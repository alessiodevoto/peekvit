import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from utils.utils import make_experiment_directory, save_state, load_state, train_only_gates_and_cls_token
from utils.logging import SimpleLogger
from peekvit.dataset import get_imagenette
from models.models import build_model
from peekvit.losses import get_loss
from utils.topology import add_residual_gates, reinit_class_tokens
from torch.utils.tensorboard import SummaryWriter



torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 160

model_class = 'VisionTransformer'

"""model_args = {
        'image_size': 160,
        'patch_size': 8,
        'num_classes': 10,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'mlp_dim': 378,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'num_registers': 0,
        'num_class_tokens': 1,
    }"""

model_args = {
        'image_size': IMAGE_SIZE,
        'patch_size': 8,
        'num_classes': 10,
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'mlp_dim': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'num_registers': 0,
        'num_class_tokens': 1,
    }


noise_args = None
gate_args = None


training_args = {
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 300,
    'eval_every': 10,
    'checkpoint_every': 30,
    'additional_loss': None ,
    'additional_loss_weights': [0,0],
    'additional_loss_args': None,
    'reinit_class_tokens': False,
}



def train(run_dir, load_from=None):

    # create run directory and logger 
    checkpoints_dir = join(run_dir, 'checkpoints')
    logger = SimpleLogger(join(run_dir, 'logs.txt'))
    logger.log(f'Experiment name: {run_dir}')
    logger.log({'training_args': training_args, 'noise_args': noise_args, 'gate_args': gate_args})
    writer = SummaryWriter(run_dir)


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
    
    # edit model topology
    if gate_args:
        model = add_residual_gates(model, gate_args)
        model_args.update(gate_args)

    if training_args['reinit_class_tokens']:
        model = reinit_class_tokens(model)
    

    # training 
    main_criterion = torch.nn.CrossEntropyLoss()
    regularization = get_loss(training_args['additional_loss'], training_args['additional_loss_args'])
    intra_weight, inter_weight = training_args['additional_loss_weights']

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    def train_epoch(model, loader, optimizer):
        model.train()
        running_loss, running_main_loss, running_intra, running_inter = 0.0, 0.0, 0.0, 0.0
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            main_loss = main_criterion(out, labels) 
            intra_reg, inter_reg  = regularization(model)
            loss = main_loss + intra_reg * intra_weight + inter_reg * inter_weight
            loss.backward()
            optimizer.step()
            
            # update running losses
            running_loss += loss.detach().item()
            running_main_loss += main_loss.detach().item()
            running_intra += intra_reg.detach().item() * intra_weight
            running_inter += inter_reg.detach().item() * inter_weight

        logger.log(f'Epoch {epoch:03} Train loss: {running_loss / len(loader)}. Main loss: {running_main_loss / len(loader)}. intra: {running_intra / len(loader)}. inter: {running_inter / len(loader)}')
        writer.add_scalar('Loss/train', running_loss / len(loader), epoch)
        writer.add_scalar('Loss/main', running_main_loss / len(loader), epoch)
        writer.add_scalar('Loss/intra', running_intra / len(loader), epoch)
        writer.add_scalar('Loss/inter', running_inter / len(loader), epoch)
    
    @torch.no_grad()
    def validate_epoch(model, loader):
        model.eval()
        correct = 0
        total = 0
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            out = model(batch)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total
    

    for epoch in range(training_args['num_epochs']+1):
        train_epoch(model, train_loader, optimizer)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            acc = validate_epoch(model, val_loader)
            logger.log(f'Epoch {epoch:03} accuracy: {acc}')
            writer.add_scalar('Accuracy/val', acc, epoch)

        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, model_args, noise_args, optimizer, epoch)


def visualize_predictions(run_dir, epoch=None):

    
    # load model from last epoch or specified epoch
    last_checkpoint = training_args['num_epochs'] if training_args['num_epochs'] > training_args['checkpoint_every'] else 0
    epoch_to_load = epoch if epoch is not None else last_checkpoint
    checkpoint_path = join(run_dir, 'checkpoints', f'epoch_{epoch_to_load:03}.pth')
    model, optimizer, epoch, model_args, noise_args = load_state(checkpoint_path, model=None, optimizer=None)    
    
    images_dir = join(run_dir, 'images')

    # transform without normalization for visualization
    visualization_transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
    ])
    
    # load dataset
    # you can decide here how many images you want to visualize
    _, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT, test_transform=visualization_transform)
    subset = torch.arange(0, 10, 1) 


    # visualize predictions
    from visualize import img_mask_distribution

    img_mask_distribution(model, 
                            val_dataset,
                            subset, 
                            model_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            save_dir = f'{images_dir}/epoch_{epoch_to_load}',
                            hard=T
                            )


def visualize_experts(run_dir, model=None, epoch=None):
    # load model from last epoch or specified epoch
    last_checkpoint = training_args['num_epochs'] if training_args['num_epochs'] > training_args['checkpoint_every'] else 0
    epoch_to_load = epoch if epoch is not None else last_checkpoint
    checkpoint_path = join(run_dir, 'checkpoints', f'epoch_{epoch_to_load}.pth')
    model, _, epoch, _, _ = load_state(checkpoint_path, model=model, optimizer=None)    
    
    
    images_dir = join(run_dir, 'images')
    

    # visualize predictions
    from visualize import display_expert_embeddings
    display_expert_embeddings(model, save_dir=f'{images_dir}/epoch_{last_checkpoint}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    args = parser.parse_args()
    if args.train:
        train_run_dir, exp_name = make_experiment_directory(BASE_PATH)
        train(train_run_dir, args.run_dir)
        visualize_predictions(train_run_dir)
    elif args.plot:
        run_dir = args.run_dir
        visualize_predictions(run_dir, epoch=args.epoch)
        # visualize_experts(run_dir)
    

