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
from utils.adapters import from_vit_to_residual_vit

torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_class = 'ResidualVisionTransformer'

"""model_args = {
        'image_size': 160,
        'patch_size': 8,
        'num_classes': 10,
        'hidden_dim': 96,
        'num_layers': 4,
        'num_heads': 8,
        'mlp_dim': 128,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'num_registers': 0,
        'num_class_tokens': 1,
    }"""


gate_args = {
    'residual_layers': ['attention+mlp', 'attention+mlp', 'attention+mlp', 'attention+mlp'],
    'gate_temp': 1,
    'add_input': False,
    'gate_type': 'sigmoid',
}



# noise_args = None
# gate_args = None


training_args = {
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'eval_every': 5,
    'checkpoint_every': 10,
    'additional_loss': 'solo_mse',
    'additional_loss_weights': [1, 0],
    'additional_loss_args': {'budget':0.65},
    'reinit_class_tokens': True,
}



def train(run_dir, load_from=None):

    # create run directory and logger 
    checkpoints_dir = join(run_dir, 'checkpoints')
    logger = SimpleLogger(join(run_dir, 'logs.txt'))
    logger.log(f'Experiment name: {run_dir}')
    # logger.log(noise_args)
    logger.log(gate_args)
    logger.log(training_args)


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
        checkpoint_model_class = torch.load(load_from)['model_class']
        if checkpoint_model_class == 'VisionTransformer':
            model, model_args = from_vit_to_residual_vit(load_from, gate_args)
        elif checkpoint_model_class == 'ResidualVisionTransformer':
            model, _, _, model_args, _ = load_state(load_from, model=None, optimizer=None)
    else:
        model = build_model(model_class, model_args, noise_args=None)
    
 
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
        model = train_only_gates_and_cls_token(model, verbose=False)
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
            visualize_predictions_in_training(model, val_dataset, torch.arange(0, 10, 1), epoch, T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), f'{run_dir}/images/epoch_{epoch}', hard=True)
            logger.log(f'Epoch {epoch:03} accuracy: {acc}')

        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, model_args, None, optimizer, epoch)


def visualize_predictions(run_dir, epoch=None):

    
    # load model from last epoch or specified epoch
    last_checkpoint = training_args['num_epochs'] if training_args['num_epochs'] > training_args['checkpoint_every'] else 0
    epoch_to_load = epoch if epoch is not None else last_checkpoint
    checkpoint_path = join(run_dir, 'checkpoints', f'epoch_{epoch_to_load:03}.pth')
    model, optimizer, epoch, model_args, noise_args = load_state(checkpoint_path, model=None, optimizer=None)    
    
    images_dir = join(run_dir, 'images')

    # transform without normalization for visualization
    visualization_transform = T.Compose([
        T.Resize(160),
        T.CenterCrop(160),
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
                            transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            save_dir = f'{images_dir}/epoch_{epoch_to_load}',
                            hard=True
                            )


def visualize_predictions_in_training(model, dataset, subset, epoch, transform, save_dir, hard=False):
    from visualize import img_mask_distribution
    img_mask_distribution(model, 
                        dataset,
                        subset, 
                        transform = transform,
                        save_dir = save_dir,
                        hard=hard
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
        train_run_dir = make_experiment_directory(BASE_PATH)
        train(train_run_dir, args.run_dir)
        visualize_predictions(train_run_dir)
    elif args.plot:
        run_dir = args.run_dir
        visualize_predictions(run_dir, epoch=args.epoch)
        # visualize_experts(run_dir)
    

