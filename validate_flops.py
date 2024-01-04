import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from utils.utils import make_experiment_directory, load_state, add_noise
from utils.logging import SimpleLogger, WandbLogger
from utils.flops_count import compute_flops
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
    'eval_batch_size': 64, # this must 1 for counting flops
    'num_workers': 4,
    'x_axis': 'flops', # 'budget' or 'flops'
    'save_images_locally': True,
    'save_images_to_wandb': False,
}


@torch.no_grad()
def validate_flops(run_dir, load_from=None, epoch=None, budgets=None):


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
    logger.log({'model_args': model_args})
    model = model.to(device)
    model.eval()

    
    accs, flops = [], []
    for budget in budgets:
        
        # compute accuracy given budget
        correct = 0
        total = 0
        for batch, labels in tqdm(val_loader, desc=f'Validating epoch {epoch} with budget {budget}'):
            batch, labels = batch.to(device), labels.to(device)
            if hasattr(model, 'set_budget'):
                model.set_budget(budget)
            out = model(batch)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
        logger.log({f'val_accuracy/budget_{budget}': acc})
        accs.append(acc)


        # compute flops given budget
        avg_flops = 0
        for batch, labels in tqdm(val_loader, desc=f'Counting flops for epoch {epoch} with budget {budget}'):
            batch, labels = batch.to(device), labels.to(device)
            if hasattr(model, 'set_budget'):
                model.set_budget(budget)
            num_flops, num_params = compute_flops(
                model, 
                batch,
                as_strings=False,
                verbose=False,
                print_per_layer_stat=False,
                flops_units='Mac'
                )
            avg_flops += num_flops
        avg_flops /= len(val_loader)
        flops.append(avg_flops)

        # avg sparsity 
        avg_sparsity = 0
        for name, module in model.named_modules():
            if hasattr(module, 'avg_sparsity'):
                avg_sparsity += module.avg_sparsity 
                module.avg_sparsity = 0
        logger.log('Average sparsity: ' + str(avg_sparsity / len(list(model.named_modules()))))

    logger.log({'flops': flops})
    # log accuracy vs budget
    from visualize import plot_budget_vs_acc
    x_axis = budgets if validation_args['x_axis'] == 'budget' else flops
    fig = plot_budget_vs_acc(x_axis, accs, epoch=epoch, save_dir=f'{run_dir}/images/epoch_{epoch}_{validation_args["x_axis"]}' if validation_args['save_images_locally'] else None)
    if validation_args['save_images_to_wandb']:
        logger.log({f'val_accuracy_vs_{validation_args["x_axis"]}': fig})




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--budgets', nargs='+', required=True)

    args = parser.parse_args()
    store_to = join(args.load_from, 'eval')

    validate_flops(store_to, 
             load_from=args.load_from, 
             epoch=args.epoch, 
             budgets=[float(b) for b in args.budgets]
             )
    

