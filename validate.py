import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from utils.utils import make_experiment_directory, load_state
from utils.logging import SimpleLogger
from peekvit.dataset import get_imagenette
from dataset import IMAGENETTE_DENORMALIZE_TRANSFORM


torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'
BASE_PATH = '/home/aledev/projects/peekvit-workspace/peekvit/runs' 

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




validation_args = {
    'eval_batch_size': 64,
    'num_workers': 4,
}

def validate(run_dir, load_from=None, epoch=None, budgets=None):

    # create run directory and logger 
    logger = SimpleLogger(join(run_dir, 'val_logs.txt'))
    logger.log(f'Experiment name: {run_dir}')


    # dataset and dataloader
    _, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=validation_args['eval_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    
    # get last checkpoint in the load_from directory
    load_from = join(load_from, 'checkpoints')
    last_checkpoint = f'epoch_{epoch}.pth' if epoch else sorted(os.listdir(load_from))[-1]
    load_from = join(load_from, last_checkpoint)
    logger.log(f'Loading model from {load_from}')
    
    model, _, epoch, model_args, _ = load_state(load_from, model=None, optimizer=None)
    model = model.to(device)


    # validate      
    @torch.no_grad()
    def validate_epoch(model, loader, budget=None):
        print('Setting budget to', budget)
        model.set_budget(float(budget))
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
    

    for budget in budgets:
        acc = validate_epoch(model, val_loader, budget=budget)
        logger.log(f'Budget: {budget} --> Accuracy: {acc}')
        visualize_predictions_in_training(model, val_dataset, torch.arange(0, 4000, 400), epoch, None, IMAGENETTE_DENORMALIZE_TRANSFORM, f'{run_dir}/images/epoch_{epoch}_budget_{budget}', hard=True)
   




def visualize_predictions(run_dir, epoch=None):

    
    # load model from last epoch or specified epoch
    last_checkpoint = validation_args['num_epochs'] if validation_args['num_epochs'] > validation_args['checkpoint_every'] else 0
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
    subset = torch.arange(0, 4000, 400) 


    # visualize predictions
    from visualize import img_mask_distribution

    img_mask_distribution(model, 
                            val_dataset,
                            subset, 
                            model_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            visualization_transform = None,
                            save_dir = f'{images_dir}/epoch_{epoch_to_load}',
                            hard=False
                            )


def visualize_predictions_in_training(model, dataset, subset, epoch, transform, visualization_transform, save_dir, hard=False):
    from visualize import img_mask_distribution
    img_mask_distribution(model, 
                        dataset,
                        subset, 
                        model_transform = transform,
                        visualization_transform = visualization_transform,
                        save_dir = save_dir,
                        hard=hard
                        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--budgets', nargs='+', required=True)

    args = parser.parse_args()

    run_dir = make_experiment_directory(BASE_PATH)
    validate(run_dir, load_from=args.run_dir, epoch=args.epoch, budgets=args.budgets)
    

