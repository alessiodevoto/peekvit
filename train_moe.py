import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peekvit.dataset import get_imagenette
from torchvision import transforms as T
from torch.utils.data import DataLoader
from models.moe_model import VisionTransformerMoE
import torch
from tqdm import tqdm

from os.path import join
from utils import SimpleLogger
from datetime import datetime

# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 1
eval_every = 5
checkpoint_every = 5

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
        'mlp_moes': [1,1,1,1],
        'attn_moes': [1,1,1,4]
    }

def make_experiment_name():
    now = datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    return join(BASE_PATH, formatted_date_time)


def train(run_dir):

    train_transform = T.Compose([
        T.RandomResizedCrop(160),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = T.Compose([
        T.Resize(160),
        T.CenterCrop(160),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset, val_dataset = get_imagenette(root=DATASET_ROOT, train_transform=train_transform, test_transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = VisionTransformerMoE(**model_args)
    
    checkpoint_path = join(run_dir, 'checkpoints')
    logger = SimpleLogger(join(run_dir, 'logs.txt'))
    logger.log(f'Experiment name: {run_dir}')
    logger.log(model_args)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_epoch(model, loader, optimizer):
        model.train()
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
    
    @torch.no_grad()
    def validate(model, loader):
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
     
    for epoch in range(num_epochs+1):
        train_epoch(model, train_loader, optimizer)
        if epoch % eval_every == 0:
            acc = validate(model, val_loader)
            logger.log(f'Epoch {epoch} accuracy: {acc}')

        if epoch % checkpoint_every == 0:
            # save
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), f'{checkpoint_path}/epoch_{epoch}.pth')



def visualize_predictions(run_dir):
    # load model
    last_checkpoint = num_epochs if num_epochs > checkpoint_every else 0
    model = VisionTransformerMoE(**model_args)
    model.to(device)
    checkpoint_path = join(run_dir, 'checkpoints')
    images_path = join(run_dir, 'images')
    model.load_state_dict(torch.load(f'{checkpoint_path}/epoch_{last_checkpoint}.pth'))
    
    # this is a transform that undoes the normalization for visualization
    unnormalized_transform = T.Compose([
        T.Resize(160),
        T.CenterCrop(160),
        T.ToTensor(),
    ])
    
    # load dataset
    # you can decide here how many images you want to visualize
    _, val_dataset = get_imagenette(root=DATASET_ROOT, test_transform=unnormalized_transform)
    subset = torch.arange(0, 4000, 400) 


    # visualize predictions
    from visualize import img_expert_distribution
    img_expert_distribution(model, 
                            val_dataset,
                            subset, 
                            transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            save_dir= f'{images_path}/expert_distribution_{last_checkpoint}')


def visualize_experts(run_dir):
    # load model
    last_checkpoint = num_epochs if num_epochs > checkpoint_every else 0
    model = VisionTransformerMoE(**model_args)
    model.to(device)
    checkpoint_path = join(run_dir, 'checkpoints')
    images_path = join(run_dir, 'images')
    model.load_state_dict(torch.load(f'{checkpoint_path}/epoch_{last_checkpoint}.pth'))
    

    # visualize predictions
    from visualize import display_expert_embeddings
    display_expert_embeddings(model, 
                            save_dir=f'{images_path}/expert_distribution_{last_checkpoint}')


if __name__ == '__main__':
    run_dir = '/home/aledev/projects/peekvit-workspace/peekvit/runs/2023_12_13_11_07_02' #make_experiment_name() # if you just want to visualize some images, set this to the path of a run
    # train(run_dir)
    visualize_predictions(run_dir)
    visualize_experts(run_dir)

