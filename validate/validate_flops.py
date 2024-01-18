import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from peekvit.utils.utils import load_state
from peekvit.utils.logging import SimpleLogger, WandbLogger
from peekvit.utils.flops_count import compute_flops
from peekvit.utils.visualize import plot_budget_vs_acc, plot_budget_vs_sparsity
from peekvit.data.imagenette import get_imagenette
from peekvit.data.imagenette import IMAGENETTE_DENORMALIZE_TRANSFORM

torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




@torch.no_grad()
def validate_flops(
    run_dir, 
    load_from=None, 
    epoch=None, 
    budgets=None, 
    n_exit=-1,
    save_images_locally: bool = True,
    save_images_to_wandb: bool = False,
    eval_batch_size: int = 64,
    image_size: int = 160
    ):


    logger = SimpleLogger(join(run_dir, 'val_logs.txt'))
    logger.log(f'Experiment name: {run_dir}')


    # dataset and dataloader
    _, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT, image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)
    
    # get last checkpoint in the load_from directory
    load_from = join(load_from, 'checkpoints')
    last_checkpoint = f'epoch_{epoch}.pth' if epoch else sorted(os.listdir(load_from))[-1]
    load_from = join(load_from, last_checkpoint)
    logger.log(f'Loading model from {load_from}')
    
    model, _, epoch, model_args, _ = load_state(load_from, model=None, optimizer=None)
    logger.log({'model_args': model_args})
    model = model.to(device)
    model.eval()

    
    accs, flops, sparsities = [], [], []
    for budget in budgets:
        
        # compute accuracy given budget
        correct = 0
        total = 0
        for batch, labels in tqdm(val_loader, desc=f'Validating epoch {epoch} with budget {budget}'):
            batch, labels = batch.to(device), labels.to(device)
            if hasattr(model, 'set_budget'):
                model.set_budget(budget)
            out = model(batch)
            # TODO explain why we are taking the last output (it's early exit)
            _, predicted = torch.max(out.data, 1) if isinstance(out, torch.Tensor) else torch.max(out[n_exit].data, 1)
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
        num_masking_modules = 0
        for name, module in model.named_modules():
            if hasattr(module, 'avg_sparsity'):
                avg_sparsity += module.avg_sparsity.detach().cpu().item()
                module.avg_sparsity = 0
                num_masking_modules += 1
        avg_sparsity = avg_sparsity / (len(list(model.named_modules()) * eval_batch_size * num_masking_modules))
        sparsities.append(avg_sparsity)
        logger.log('Average sparsity: ' + str(avg_sparsity))
    
    logger.log({'flops': flops})
    # log accuracy vs budget
    fig_budget = plot_budget_vs_acc(budgets, accs, epoch=epoch, save_dir=f'{run_dir}/images/epoch_{epoch}_budgets' if save_images_locally else None)
    fig_flops = plot_budget_vs_acc(flops, accs, epoch=epoch, save_dir=f'{run_dir}/images/epoch_{epoch}_flops' if save_images_locally else None)
    fig_sparsity = plot_budget_vs_sparsity(flops, sparsities, epoch=epoch, save_dir=f'{run_dir}/images/epoch_{epoch}_sparsity' if save_images_locally else None)

    if save_images_to_wandb:
        logger.log({'val_accuracy_vs_flops': fig_flops, 'val_accuracy_vs_budget': fig_budget, 'val_accuracy_vs_sparsity': fig_sparsity})




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, default=None, help='Path to the experiment directory containing the checkpoint')
    parser.add_argument('--epoch', type=str, default=None, help='Epoch to load from. If None, the last checkpoint is loaded.')
    parser.add_argument('--image_size', type=int, default=160, help='Image size to use for the dataset.')
    parser.add_argument('--budgets', nargs='+', required=True, help='Budgets to validate with.')
    parser.add_argument('--n_exit', type=int, default=-1, help='Early exit to validate with. If -1, the last exit is used.')
    parser.add_argument('--store_locally', action='store_true', help='If true, images are saved locally.')
    parser.add_argument('--store_wandb', action='store_true', help='If true, images are saved to Wandb.')
    parser.add_argument('--eval_bs', type=int, default=64, help='Evaluation batch size.')


    args = parser.parse_args()

    if not args.store_locally and not args.store_wandb:
        raise ValueError('At least one of store_locally or store_wandb must be true.')

    # create directory to store results, in the load_from directory
    store_to = join(args.load_from, 'eval')

    validate_flops(store_to, 
            load_from=args.load_from, 
            epoch=args.epoch, 
            budgets=[float(b) for b in args.budgets],
            n_exit=args.n_exit,
            save_images_locally=args.store_locally,
            save_images_to_wandb=args.store_wandb,
            eval_batch_size=args.eval_bs,
            image_size=args.image_size
            )
    

