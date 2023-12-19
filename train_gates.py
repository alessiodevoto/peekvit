import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from utils.utils import make_experiment_directory, save_state, load_state, add_residual_gates, train_only_gates_and_cls_token, reinit_class_tokens
from utils.logging import SimpleLogger
from peekvit.dataset import get_imagenette
from models.models import build_model
from peekvit.losses import get_loss


torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model_class = 'VisionTransformerMoE'
model_class = 'ResidualVisionTransformer'
# model_class = 'VisionTransformer'

model_args = {
        'image_size': 160,
        'patch_size': 8,
        'num_classes': 10,
        'hidden_dim': 96,
        'num_layers': 4,
        'num_heads': 8,
        'mlp_dim': 128,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        # 'mlp_moes': [1,1,1,2],
        # 'attn_moes': [1,1,1,1],
        'residual_layers': ['attention+mlp', 'attention+mlp', 'attention+mlp', 'attention+mlp'],
        'num_registers': 0,
        'threshold': 0.5,
        'num_class_tokens': 1,
        'add_input': True,
        'gate_type': 'sigmoid',
    }

training_args = {
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'lr': 1e-4,
    'num_epochs': 100,
    'eval_every': 5,
    'checkpoint_every': 10,
    'additional_loss': 'solo_l1',
    'additional_loss_weight':2,
    'additional_loss_args': {}
}

gate_args = {
    'skip': 'attention+mlp',
    'temp': 0.1,
    'add_input': True,
    'gate_type': 'sigmoid',
}


"""noise_args = {
    'noise_type': 'gaussian',
    'snr': 1,
    'std': None,
    'layers': [2]
}"""

noise_args = None



def train(run_dir, load_from=None):

    checkpoints_dir = join(run_dir, 'checkpoints')
    logger = SimpleLogger(join(run_dir, 'logs.txt'))
    logger.log(f'Experiment name: {run_dir}')
    logger.log(noise_args)
    logger.log(gate_args)
    logger.log(training_args)

    train_dataset, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=training_args['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args['eval_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    
    # get last checkpoint in the load_from directory
    load_from = join(load_from, 'checkpoints')
    last_checkpoint = sorted(os.listdir(load_from))[-1]
    load_from = join(load_from, last_checkpoint)
    logger.log(f'Loading model from {load_from}')
    model, _, epoch = load_state(load_from, model=None, optimizer=None)
    model = add_residual_gates(model, gate_args)
    model = reinit_class_tokens(model)
   
    main_criterion = torch.nn.CrossEntropyLoss()
    regularization = get_loss(training_args['additional_loss'], {})
    regularization_weight = training_args['additional_loss_weight']

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    def train_epoch(model, loader, optimizer):
        model.train()
        model = train_only_gates_and_cls_token(model)
        running_loss, running_main_loss, running_entr1, running_entr2 = 0.0, 0.0, 0.0, 0.0
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            main_loss = main_criterion(out, labels)
            entr1, entr2  = regularization(model)
            # loss = main_loss + reg * regularization_weight + entr * +3
            # loss = reg * regularization_weight #+ entr * 0.0001
            loss = main_loss + entr1 * 5 + entr2 * - 2
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            running_main_loss += main_loss.detach().item()
            running_entr2 += entr2.detach().item() * - 2
            running_entr1 += entr1.detach().item() * 5

        logger.log(f'Epoch {epoch:03} Train loss: {running_loss / len(loader)}. Main loss: {running_main_loss / len(loader)}. l2: {running_entr1 / len(loader)}. entr2: {running_entr2 / len(loader)}')
        
    
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

        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, model_args, noise_args, optimizer, epoch)


def visualize_predictions(run_dir, epoch=None):

    model_args = {
        'image_size': 160,
        'patch_size': 8,
        'num_classes': 10,
        'hidden_dim': 96,
        'num_layers': 4,
        'num_heads': 8,
        'mlp_dim': 128,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        # 'mlp_moes': [1,1,1,2],
        # 'attn_moes': [1,1,1,1],
        'residual_layers': ['attention+mlp', 'attention+mlp', 'attention+mlp', 'attention+mlp'],
        'num_registers': 0,
        'threshold': 0.5,
        'num_class_tokens': 1,
        'add_input': True,
        'gate_type': 'sigmoid',
    }
    from models.residualvit import ResidualVisionTransformer
    model = ResidualVisionTransformer(**model_args)
    
    # load model from last epoch or specified epoch
    last_checkpoint = training_args['num_epochs'] if training_args['num_epochs'] > training_args['checkpoint_every'] else 0
    epoch_to_load = epoch if epoch is not None else last_checkpoint
    checkpoint_path = join(run_dir, 'checkpoints', f'epoch_{epoch_to_load:03}.pth')
    model, optimizer, epoch = load_state(checkpoint_path, model=model, optimizer=None)    
    
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
    subset = torch.arange(0, 4000, 250) 


    # visualize predictions
    from visualize import img_expert_distribution, img_mask_distribution

    img_mask_distribution(model, 
                            val_dataset,
                            subset, 
                            transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            save_dir = f'{images_dir}/epoch_{epoch_to_load}',
                            hard=True
                            )


def visualize_experts(run_dir, model=None, epoch=None):
    # load model from last epoch or specified epoch
    last_checkpoint = training_args['num_epochs'] if training_args['num_epochs'] > training_args['checkpoint_every'] else 0
    epoch_to_load = epoch if epoch is not None else last_checkpoint
    checkpoint_path = join(run_dir, 'checkpoints', f'epoch_{epoch_to_load}.pth')
    model, optimizer, epoch = load_state(checkpoint_path, model=model, optimizer=None)    
    
    
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
    

