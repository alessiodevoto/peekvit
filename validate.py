import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from utils.utils import make_experiment_directory, load_state
from utils.logging import SimpleLogger, WandbLogger
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
    'save_images_locally': True,
    'save_images_to_wandb': False,
    'masks': False,
}

@torch.no_grad()
def validate(run_dir, load_from=None, epoch=None, budgets=None):

    print(budgets)

    # logging
    if validation_args['save_images_to_wandb']:
        logger = WandbLogger(wandb_run=exp_name, wandb_run_dir=run_dir)
    else:
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
    model.eval()
    
    accs = []
    for budget in budgets:
        
        # compute accuracy given budget
        correct = 0
        total = 0
        for batch, labels in tqdm(val_loader, desc=f'Validating epoch {epoch} with budget {budget}'):
            batch, labels = batch.to(device), labels.to(device)
            if budget is not None:
                model.set_budget(budget)
            out = model(batch)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
        logger.log({f'val_accuracy/budget_{budget}': acc})
        accs.append(acc)


        # visualize predictions
        if validation_args['masks']:
            from visualize import img_mask_distribution
            img_mask_distribution(model, 
                        val_dataset,
                        torch.arange(0, 4000, 400), 
                        model_transform = None,
                        visualization_transform=IMAGENETTE_DENORMALIZE_TRANSFORM,
                        save_dir=f'{run_dir}/images/epoch_{epoch}_budget{budget}' if validation_args['save_images_locally'] else None,
                        hard=True,
                        budget=budget,
                        log_to_wandb=validation_args['save_images_to_wandb'],
                        )

    # log accuracy vs budget
    from visualize import plot_budget_vs_acc
    print(accs)
    print(budgets)
    fig = plot_budget_vs_acc(budgets, accs, epoch=epoch, save_dir=f'{run_dir}/images/epoch_{epoch}_budgets' if validation_args['save_images_locally'] else None)
    if validation_args['save_images_to_wandb']:
        logger.log({'val_accuracy_vs_budget': fig})




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--budgets', nargs='+', required=True)

    args = parser.parse_args()

    run_dir, exp_name = make_experiment_directory(BASE_PATH)
    validate(run_dir, load_from=args.run_dir, epoch=args.epoch, budgets=[float(b) for b in args.budgets])
    

