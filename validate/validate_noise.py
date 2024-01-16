from collections import defaultdict
import os, sys

from utils.flops_count import compute_flops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse



from peekvit.utils.utils import make_experiment_directory, load_state, add_noise
from peekvit.utils.logging import SimpleLogger, WandbLogger
from peekvit.data.dataset import get_imagenette
from peekvit.utils.visualize import plot_model_budget_vs_noise_vs_acc, plot_model_noise_vs_budget_vs_acc

from peekvit.data.dataset import IMAGENETTE_DENORMALIZE_TRANSFORM


torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





@torch.no_grad()
def validate_with_noise(
    run_dir, 
    load_from=None, 
    epoch=None,
    noise_type=None,
    noise_layer=None,
    noise_vals=None, 
    budgets=None,
    eval_batch_size: int = 64
    ):

    
    logger = SimpleLogger(join(run_dir, 'val_logs.txt'))
    logger.log(f'Experiment name: {run_dir}')


    # dataset and dataloader
    _, val_dataset, _, _ = get_imagenette(root=DATASET_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)

    
    # get checkpoint and load model
    load_from = join(load_from, 'checkpoints')
    last_checkpoint = f'epoch_{epoch}.pth' if epoch else sorted(os.listdir(load_from))[-1]
    load_from = join(load_from, last_checkpoint)
    logger.log(f'Loading model from {load_from}')
    
    model, _, epoch, model_args, _ = load_state(load_from, model=None, optimizer=None)
    model = model.to(device)
    model.eval()

    # sanity check that model has budget
    if not hasattr(model, 'set_budget'):
        print('Model does not have budget, setting to None')
        budgets = [0.0]
    
    # add noise module
    noise_module = add_noise(
                model, 
                noise_type=noise_type, 
                layer=noise_layer)

    # validate
    # this will be a dict of dicts. {budget : {noise_value : accuracy}}
    results_per_budget = defaultdict(dict)
    results_per_flops = defaultdict(dict)
    
    for budget in budgets:

        print(f'Validating with budget {budget}')
        results_per_budget[budget] = {}

        if hasattr(model, 'set_budget'):
            model.set_budget(budget)
            
        accs = []
        
        for val in noise_vals:

            print(f'Validating with {noise_type} noise with value {val}')

            noise_module.set_value(val)
            
            # compute accuracy given budget
            correct = 0
            total = 0
            for batch, labels in tqdm(val_loader, desc=f'Validating epoch {epoch} with budget {budget}'):
                batch, labels = batch.to(device), labels.to(device)
                # print(model.current_budget)
                out = model(batch)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct / total
            logger.log({f'val_accuracy/budget_{budget}': acc})
            accs.append(acc)

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
            

            results_per_budget[budget][val] = acc
            results_per_flops[avg_flops][val] = acc

    logger.log({'flops': results_per_flops, 'budget': results_per_budget})
    
    return results_per_budget, results_per_flops


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program with two arguments.')
    parser.add_argument('--load_from', type=str, default=None, help='Path to the experiment directory containing the checkpoint')
    parser.add_argument('--epoch', type=str, default=None, help='Epoch to load from. If None, the last checkpoint is loaded.')
    parser.add_argument('--budgets', nargs='+', required=True, help='Budgets to validate with.')
    parser.add_argument('--probs', nargs='*', default=None, help='Probabilities of token dropping noise to validate with.')
    parser.add_argument('--snrs', nargs='*', default=None, help='SNRs of gaussian noise to validate with.')
    parser.add_argument('--noise_layer', type=int, default=2, help='Layer to add noise to.')
    parser.add_argument('--store_locally', action='store_true', help='If true, images are saved locally.')
    parser.add_argument('--store_wandb', action='store_true', help='If true, images are saved to Wandb.')
    parser.add_argument('--eval_bs', type=int, default=64, help='Evaluation batch size.')

    args = parser.parse_args()

    if not args.store_locally and not args.store_wandb:
        raise ValueError('At least one of store_locally or store_wandb must be true.')

    # create directory to store results, in the load_from directory
    store_to = join(args.load_from, 'eval')

    # convert args to floats
    budgets = [float(b) for b in args.budgets] if args.budgets else [None]
    probs = [float(p) for p in args.probs] if args.probs else None
    snrs = [float(s) for s in args.snrs] if args.snrs else None

    if probs is not None and snrs is not None:
        raise ValueError('Cannot specify both probs and snrs')
    
    
    all_results_per_budget = {}
    all_results_per_flops = {}

    for load_from in args.load_from:
        budget_results, flops_results = validate_with_noise(
                        store_to, 
                        load_from=load_from, 
                        epoch=args.epoch, 
                        noise_layer=args.noise_layer,
                        budgets=budgets,
                        noise_type= 'gaussian' if args.snrs else 'token_drop',
                        noise_vals=snrs if args.snrs else probs,
                        )
        all_results_per_budget[load_from] = budget_results
        all_results_per_flops[load_from] = flops_results
    
    

    budet_vs_acc = plot_model_budget_vs_noise_vs_acc(all_results_per_budget, save_dir=f'{store_to}/images/' if args.store_locally else None)
    flop_vs_acc = plot_model_noise_vs_budget_vs_acc(all_results_per_flops, additional_x_labels=budgets, save_dir=f'{store_to}/images/' if args.store_locally else None)

    if args.store_wandb:
        logger = WandbLogger(wandb_run_dir=store_to)
        logger.log({'val_accuracy_vs_budget': budet_vs_acc, 'val_accuracy_vs_flops': flop_vs_acc})
