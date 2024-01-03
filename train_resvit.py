import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from utils.utils import make_experiment_directory, save_state, load_state, train_only_these_params
from utils.logging import WandbLogger, SimpleLogger
from models.models import build_model
from utils.topology import add_residual_gates, reinit_class_tokens
from utils.adapters import from_vit_to_residual_vit
from utils.utils import add_noise
from peekvit.dataset import IMAGENETTE_DENORMALIZE_TRANSFORM
from peekvit.dataset import get_imagenette
from peekvit.losses import get_loss

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
    #'residual_layers': ['attention+mlp', 'attention+mlp', 'attention+mlp', 'attention+mlp'],
    'residual_layers': ['attention+mlp', 'attention+mlp', None, None],
    'gate_temp': 1,
    'add_input': False,
    'gate_type': 'sigmoid',
    'gate_threshold': 0.5,
    'gate_bias': -0.5,
    'add_budget_token': 'learnable' # this can be either True (sample bugdet from a uniform distribution) or a float (constant budget) or list of floats (sample budget from this list)
}

training_args = {
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 200,
    'eval_every': 10,
    'checkpoint_every': 20,
    'additional_loss': 'solo_mse',
    'additional_loss_weights': [0.05, 0],
    'additional_loss_args': {'budget': 'budget_token', 'strict': True},
    'reinit_class_tokens': True,
    'wandb': False,
    'save_images_locally': True,
    'save_images_to_wandb': False,
    }

"""noise_args = {
    'layer': 2,
    'noise_type': 'gaussian',
    'snr': 200,
}"""

noise_args = {}



VALIDATE_BUDGETS = [None] if training_args['additional_loss_args']['budget'] != 'budget_token' else [0.25, 0.85]


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
            model, model_args = from_vit_to_residual_vit(load_from, gate_args)
        elif checkpoint_model_class == 'ResidualVisionTransformer':
            model, _, _, model_args, _ = load_state(load_from, model=None, optimizer=None)
    else:
        model = build_model(model_class, model_args)
    
    # logging
    if training_args['wandb']:
        logger = WandbLogger(entity=wandb_entity, project=wandb_project, config=training_args | gate_args | model_args | noise_args, wandb_run=exp_name, wandb_run_dir=run_dir)
    else:
        logger = SimpleLogger(join(run_dir, 'logs.txt'))
        logger.log({'model_class': model_class, 'model_args': model_args, 'gate_args': gate_args, 'noise_args': noise_args, 'training_args': training_args})
    
    # adjust model
    if training_args['reinit_class_tokens']:
        model = reinit_class_tokens(model)
    
    # add noise
    if noise_args != {}:
        model = add_noise(model, **noise_args)
        print(model)
    

    # training 
    main_criterion = torch.nn.CrossEntropyLoss()
    regularization = get_loss(training_args['additional_loss'], training_args['additional_loss_args'])
    intra_weight, inter_weight = training_args['additional_loss_weights']

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    # budget for regularization
    # if budget is a float, it is used as a constant, else we use the budget token which is the last token in the sequence
    training_budget = training_args['additional_loss_args']['budget']
    if training_budget == 'budget_token':
        get_training_budget = lambda model : model.current_budget
    else:
        # training budget is a float
        get_training_budget = lambda model : training_budget

    def train_epoch(model, loader, optimizer, epoch=None):
        model.train()
        model = train_only_these_params(model, verbose=False, params_list=['gate', 'budget', 'class', 'head', 'threshold'])
        running_loss, running_main_loss, running_intra, running_inter = 0.0, 0.0, 0.0, 0.0
        for batch, labels in tqdm(loader, desc=f'Training epoch {epoch}'):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            main_loss = main_criterion(out, labels)    
            intra_reg, inter_reg = regularization(model, budget=get_training_budget(model))
            loss = main_loss + intra_reg * intra_weight + inter_reg * inter_weight
            loss.backward()
            optimizer.step()
            
            # update running losses
            running_loss += loss.detach().item()
            running_main_loss += main_loss.detach().item()
            running_intra += intra_reg.detach().item() * intra_weight
            running_inter += inter_reg.detach().item() * inter_weight

        # logger.log(f'Epoch {epoch:03} Train loss: {running_loss / len(loader)}. Main loss: {running_main_loss / len(loader)}. intra: {running_intra / len(loader)}. inter: {running_inter / len(loader)}')
        logger.log({'epoch':epoch, 'train_loss': running_loss / len(loader), 'train_main_loss': running_main_loss / len(loader), 'train_intra': running_intra / len(loader), 'train_inter': running_inter / len(loader)})


    @torch.no_grad()
    def validate_epoch(model, loader, budgets=None, epoch=None):
        model.eval()
        
        accs = []
        for budget in budgets:
            
            # compute accuracy given budget
            correct = 0
            total = 0
            for batch, labels in tqdm(loader, desc=f'Validating epoch {epoch} with budget {budget}'):
                batch, labels = batch.to(device), labels.to(device)
                if budget is not None:
                    model.set_budget(float(budget))
                out = model(batch)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct / total
            logger.log({f'val_accuracy/budget_{budget}': acc})
            accs.append(acc)


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
        from visualize import plot_budget_vs_acc
        val_accuracy_vs_budget_fig = plot_budget_vs_acc(budgets, accs, epoch=epoch, save_dir=None)
        logger.log({'val_accuracy_vs_budget': val_accuracy_vs_budget_fig})
            
    

    for epoch in range(training_args['num_epochs']+1):
        train_epoch(model, train_loader, optimizer, epoch=epoch)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            validate_epoch(model, val_loader, VALIDATE_BUDGETS, epoch=epoch)
            
        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, model_args, None, optimizer, epoch)
     




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    args = parser.parse_args()
    if args.train:
        train_run_dir, exp_name = make_experiment_directory(BASE_PATH)
        train(run_dir=train_run_dir, load_from=args.run_dir, exp_name=exp_name)

        
    

