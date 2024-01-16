import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from peekvit.utils.utils import make_experiment_directory, save_state, load_state
from peekvit.utils.logging import SimpleLogger, WandbLogger
from peekvit.data.dataset import get_imagenette
from peekvit.models.models import build_model

from torchvision.models.vision_transformer import ViT_B_16_Weights


torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 224

model_class = 'VisionTransformer'

# WANDB
wandb_entity = 'aledevo'
wandb_project = 'peekvit'


"""model_args = {
        'image_size': IMAGE_SIZE,
        'patch_size': 16,
        'num_classes': 10,
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'mlp_dim': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'num_registers': 0,
        'num_class_tokens': 1,
    }"""


model_args = {
    'image_size':224,
    'patch_size':16,
    'num_layers':12,
    'num_heads':12,
    'hidden_dim':768,
    'mlp_dim':3072,
    'num_classes': 10,
    'torch_pretrained_weights': ViT_B_16_Weights['IMAGENET1K_V1'].get_state_dict(),
}


noise_args = {}
gate_args = {}


training_args = {
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'lr': 1e-4,
    'num_epochs': 100,
    'eval_every': 1,
    'checkpoint_every': 1,
    'wandb': True,
    'save_images_locally': False,
    'save_images_to_wandb': True,
}



def train(run_dir, load_from=None):

    # create run directory and logger 
    checkpoints_dir = join(run_dir, 'checkpoints')
    logger = SimpleLogger(join(run_dir, 'logs.txt'))
    logger.log(f'Experiment name: {run_dir}')
    logger.log({'training_args': training_args, 'noise_args': noise_args, 'gate_args': gate_args})


    # dataset and dataloader
    train_dataset, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=training_args['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args['eval_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    
    # get last checkpoint in the load_from directory
    if load_from is not None:
        load_from = join(load_from, 'checkpoints')
        last_checkpoint = sorted(os.listdir(load_from))[-1]
        load_from = join(load_from, last_checkpoint)
        logger.log(f'Loading model from {load_from}')
        model, optimizer, epoch, loaded_model_args, _ = load_state(load_from, model=None, optimizer=None)
        model_args.update(loaded_model_args)
    else:
        model = build_model(model_class, model_args, noise_args)
    

    if training_args['wandb']:
        logger = WandbLogger(entity=wandb_entity, project=wandb_project, config=training_args | gate_args | model_args | noise_args, wandb_run=exp_name, wandb_run_dir=run_dir)
    else:
        logger = SimpleLogger(join(run_dir, 'logs.txt'))
        logger.log({'model_class': model_class, 'model_args': model_args, 'gate_args': gate_args, 'noise_args': noise_args, 'training_args': training_args})


    # training 
    main_criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    def train_epoch(model, loader, optimizer):
        model.train()
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            main_loss = main_criterion(out, labels) 
            loss = main_loss 
            loss.backward()
            optimizer.step()
            
            logger.log({'train/loss': main_loss.detach().item()})


    
    @torch.no_grad()
    def validate_epoch(model, loader):
        model.eval()
        correct = 0
        total = 0
        avg_val_loss = 0
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            out = model(batch)
            val_loss = main_criterion(out, labels) 
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            avg_val_loss += val_loss.detach().item()
        acc = correct / total
        avg_val_loss = avg_val_loss / len(loader)
        logger.log({'val/acc': acc, 'val/loss': avg_val_loss})
        return acc, avg_val_loss
    

    for epoch in range(training_args['num_epochs']+1):
        train_epoch(model, train_loader, optimizer)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            validate_epoch(model, val_loader)
            

        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, model_args, noise_args, optimizer, epoch)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_from', type=str, default=None)
    args = parser.parse_args()
    if args.train:
        train_run_dir, exp_name = make_experiment_directory(BASE_PATH)
        train(train_run_dir, args.load_from)

    

