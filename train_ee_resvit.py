from collections import defaultdict
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse
import random



from utils.utils import make_experiment_directory, save_state, load_state, train_only_these_params
from utils.logging import WandbLogger, SimpleLogger
from models.models import build_model
from utils.topology import reinit_class_tokens
from utils.adapters import from_vit_to_eeresidual_vit
from utils.utils import add_noise
from peekvit.dataset import IMAGENETTE_DENORMALIZE_TRANSFORM
from peekvit.dataset import get_imagenette
from peekvit.losses import get_losses

torch.manual_seed(0)



"""
Explanation of this script:
- this script is a simple example of how to train a ResidualVisionTransformer with residual gates
- the model is trained on the Imagenette dataset
- the model is trained with a cross entropy loss and a regularization loss, see `losses.py` for more details
- the ResidualVisionTransformer can be initialized from a pretrained VisionTransformer, in this case, you must provide a `--run_dir` argument. 
    If you do not provide this argument, the model is initialized randomly with the provided `model_args` and `gate_args`.

- for the budget regularization (`add_budget_token` parameter), you can choose among:
    - a constant budget, appended to the input sequence
    - a budget sampled from a uniform distribution, appended to the input sequence
    - a budget sampled from a list of budgets, appended to the input sequence
    - `learnable` budget. In this case, we sample a budget and use it to interpolate between two lernable tokens, which are appended to the input sequence.
"""




# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# WANDB
wandb_entity = 'aledevo'
wandb_project = 'peekvit'

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_class = 'ResidualVisionTransformer'

model_args = {} # we use a pretrained model, so we do not need to specify the model args

gate_args = {
    'residual_layers': ['attention+mlp', 'attention+mlp', 'attention+mlp', 'attention+mlp'],
    'gate_temp': 1,
    'add_input': False,
    'gate_type': 'sigmoid',
    'gate_threshold': 0.5,
    'gate_bias': 3,
    'add_budget_token': 'learnable' #'learnable_interpolate' # this can be either True (sample bugdet from a uniform distribution) or a float (constant budget) or list of floats (sample budget from this list)
}


training_args = {
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 200,
    'eval_every': 10,
    'checkpoint_every': 10,
    'reinit_class_tokens': True,
    'wandb': True,
    'save_images_locally': False,
    'save_images_to_wandb': True,
    }


additional_losses_args = {
    'solo_mse' : {
        'budget': 'budget_token', 
        'strict': False,
        'weight': 0.1,
        },
    }

noise_args = {}


VALIDATE_BUDGETS = [gate_args['add_budget_token']] if isinstance(gate_args['add_budget_token'], (float, list, tuple)) else [0.25, 0.85]


def train(run_dir, load_from=None, exp_name=None):

    # create run directory and logger 
    checkpoints_dir = join(run_dir, 'checkpoints')
    
    # dataset and dataloader
    train_dataset, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=training_args['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args['eval_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # get last checkpoint in the load_from directory
    if load_from is not None:
        load_from = join(load_from, 'checkpoints')
        last_checkpoint = sorted(os.listdir(load_from))[-1]
        load_from = join(load_from, last_checkpoint)
        print(f'Loading model from {load_from}')
        checkpoint_model_class = torch.load(load_from)['model_class']
        if checkpoint_model_class == 'VisionTransformer':
            model, model_args = from_vit_to_eeresidual_vit(load_from, gate_args)
        elif checkpoint_model_class == 'EEResidualVisionTransformer':
            model, _, _, model_args, _ = load_state(load_from, model=None, optimizer=None)
    else:
        model = build_model(model_class, model_args)
    
    # logging
    if training_args['wandb']:
        logger = WandbLogger(entity=wandb_entity, project=wandb_project, config=training_args | gate_args | model_args | noise_args | additional_losses_args, wandb_run=exp_name, wandb_run_dir=run_dir)
    else:
        logger = SimpleLogger(join(run_dir, 'logs.txt'))
        logger.log({'model_class': model_class, 'model_args': model_args, 'gate_args': gate_args, 'noise_args': noise_args, 'training_args': training_args})
    
    # adjust model
    if training_args['reinit_class_tokens']:
        model = reinit_class_tokens(model)
    
    # add noise
    noise_block = None
    if noise_args != {}:
        noise_range = noise_args.pop('noise_range')
        noise_block = add_noise(model, **noise_args)
        
        
    # losses
    main_criterion = torch.nn.CrossEntropyLoss()
    regs, reg_weights = get_losses(additional_losses_args)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    # budget for regularization
    # if budget is a float, it is used as a constant, else we use the budget token which is the last token in the sequence
    training_budget = gate_args['add_budget_token']
    if isinstance(training_budget, str):
        get_training_budget = lambda model : model.current_budget
    else:
        # training budget is a float
        get_training_budget = lambda _ : training_budget

    def train_epoch(model, loader, optimizer, epoch=None):
        model.train()
        model = train_only_these_params(model, verbose=False, params_list=['gate', 'budget', 'class', 'head', 'threshold'])

        for batch, labels in tqdm(loader, desc=f'Training epoch {epoch}'):
            # set noise
            if noise_block is not None:
                noise_block.set_value(random.uniform(*noise_range) if len(noise_range) == 2 else random.choice(noise_range))
                

            # forward pass
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(batch)
            
            # we hve to compute the loss for each early exit
            main_losses = {}
            for i, out in enumerate(outs):
                main_losses[f'exit:{i}'] = main_criterion(out, labels)
            
            # compute the main loss
            main_loss = torch.sum(torch.stack([x for x in main_losses.values()], dim=-1))
            

            # multiply each regularization loss by its weight
            regularization_losses = {loss_name: loss_fn(model, budget=get_training_budget(model)) for loss_name, loss_fn in regs.items()}
            for loss_name, loss_value in regularization_losses.items():
                regularization_losses[loss_name] = loss_value * reg_weights[loss_name]

            
            loss = main_loss + torch.sum(torch.stack([x for x in regularization_losses.values()], dim=-1))
            loss.backward()
            optimizer.step()
            
            logger.log(regularization_losses | main_losses | {'train_loss': loss.item(), 'train_main_loss': main_loss.item() })



    @torch.no_grad()
    def validate_epoch(model, loader, budgets=None, epoch=None):
        model.eval()
        
        if noise_block is not None:
            noise_block.set_value(0.0)

        
        for budget in budgets:
            correct_per_exit = defaultdict(int)
            # compute accuracy given budget
            total = 0
            for batch, labels in tqdm(loader, desc=f'Validating epoch {epoch} with budget {budget}'):
                batch, labels = batch.to(device), labels.to(device)
                model.set_budget(float(budget))
                outs = model(batch)
                total += labels.size(0)
                for i, out in enumerate(outs):
                    _, predicted = torch.max(out.data, 1)
                    correct_per_exit[f'exit:{i}'] += (predicted == labels).sum().item() 
                
            for i, correct in correct_per_exit.items():
                acc = correct / total
                logger.log({f'val_accuracy/budget_{budget}/{i}': acc})    
            

            # visualize predictions
            if training_args['save_images_locally'] or training_args['save_images_to_wandb']:
                from visualize import img_mask_distribution
                img_mask_distribution(model, 
                            val_dataset,
                            torch.arange(0, 4000, 400), 
                            model_transform = None,
                            visualization_transform=IMAGENETTE_DENORMALIZE_TRANSFORM,
                            save_dir=f'{run_dir}/images/epoch_{epoch}_budget{budget}' if training_args['save_images_locally'] else None,
                            hard=True,
                            budget=budget,
                            log_to_wandb=training_args['save_images_to_wandb'],
                            )

        # log accuracy vs budget
        #from visualize import plot_budget_vs_acc
        # val_accuracy_vs_budget_fig = plot_budget_vs_acc(budgets, accs, epoch=epoch, save_dir=None)
        #logger.log({'val_accuracy_vs_budget': val_accuracy_vs_budget_fig})
            
    

    for epoch in range(training_args['num_epochs']+1):
        train_epoch(model, train_loader, optimizer, epoch=epoch)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            validate_epoch(model, val_loader, VALIDATE_BUDGETS, epoch=epoch)
            
        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, model_args, noise_args, optimizer, epoch)
     



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    args = parser.parse_args()
    if args.train:
        train_run_dir, exp_name = make_experiment_directory(BASE_PATH)
        train(run_dir=train_run_dir, load_from=args.load_from, exp_name=exp_name)

        
    

