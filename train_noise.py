import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peekvit.dataset import get_imagenette
from torchvision import transforms as T
from torch.utils.data import DataLoader
from models.moevit import VisionTransformerMoE
from models.residualvit import ResidualVisionTransformer
import torch
from tqdm import tqdm

from os.path import join
from utils import SimpleLogger, make_experiment_directory
from utils import add_noise


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 15
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
        #'mlp_moes': [1,1,1,1],
        #'attn_moes': [1,1,1,1]
        'residual_layers': [True, True, False, False]
    }



def train(run_dir):

    # Load dataset with default transforms
    train_dataset, val_dataset, *_ = get_imagenette(root=DATASET_ROOT) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ResidualVisionTransformer(**model_args)
    model = add_noise(model, layer=2, noise_snr=1.0)
    
    checkpoint_path = join(run_dir, 'checkpoints')
    logger = SimpleLogger(join(run_dir, 'logs.txt'))
    logger.log(f'Experiment name: {run_dir}')
    logger.log(model_args)

    #print(model)

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
    
    for epoch in range(num_epochs+1):
        train_epoch(model, train_loader, optimizer)
        
        if epoch % eval_every == 0:
            acc = validate_epoch(model, val_loader)
            logger.log(f'Epoch {epoch} accuracy: {acc}')

        if epoch % checkpoint_every == 0:
            # save
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), f'{checkpoint_path}/epoch_{epoch}.pth')


def visualize_predictions(run_dir, epoch=None):
    
    # load model from last epoch or specified epoch
    last_checkpoint = num_epochs if num_epochs > checkpoint_every else 0
    last_checkpoint = epoch if epoch is not None else last_checkpoint

    model = ResidualVisionTransformer(**model_args)
    model = add_noise(model, layer=2, noise_snr=10.0)
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
    from visualize import img_mask_distribution
    img_mask_distribution(model, 
                            val_dataset,
                            subset, 
                            transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            save_dir= f'{images_path}/expert_distribution_{last_checkpoint}')



if __name__ == '__main__':
    run_dir = make_experiment_directory(BASE_PATH) # if you just want to visualize some images, set this to the path of a run
    train(run_dir)
    visualize_predictions(run_dir)
