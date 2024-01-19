import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any, List
from matplotlib import pyplot as plt
from collections import defaultdict


from omegaconf import DictConfig, OmegaConf
import torchmetrics
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse
from hydra.utils import instantiate
from peekvit.utils.utils import get_checkpoint_path, make_experiment_directory, load_state, add_noise
import hydra
from pprint import pprint

from peekvit.utils.utils import load_state, add_noise, make_experiment_directory
from peekvit.utils.visualize import plot_budget_and_noise_recap, plot_cumulative_budget_recap, plot_budget_recap, plot_cumulative_budget_and_noise_recap
from peekvit.utils.flops_count import compute_flops


torch.manual_seed(0)


# PATHS 
# all images, checkpoints and logs will be saved to base path in a structured way
DATASET_ROOT = '/home/aledev/projects/moe-workspace/data/imagenette'

# HYPERPARAMETERS 
# defined here as this is a quick experiment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def validate(
        model_checkpoint: str, 
        logger: Any,
        val_loader: DataLoader, 
        budgets: List,
        noise_settings: DictConfig,
        noises: List,
        device: torch.device,
        ):
    

    # if model parameters are not specified in the config file, load the model from the checkpoint
    model, _, epoch, _, _ = load_state(model_checkpoint, model=None, strict=True)
    model.eval()
    model.to(device)
    

    # sanity check that model has budget
    if not hasattr(model, 'set_budget'):
        print('Model does not have budget, setting to None')
        budgets = budgets or [1.0]
    
    if budgets is None or len(budgets) == 0:
        print('Budgets not specified, setting to None')
        budgets = [1.0]
    
    # add noise module
    noise_module = None
    noise_vals = [None]
    noise_type = None
    if noise_settings is not None and noise_settings != {}:
        noise_type = noise_settings.noise_type
        noise_module = add_noise(noise_type=noise_settings.noise_type, layer=noise_settings.layer, model=model)
        noise_vals = noises 
        
    
    # validate
    # this will be a dict of dicts. {budget_or_flops : {noise_value : accuracy}}
    results_per_budget = defaultdict(dict)
    results_per_flops = defaultdict(dict)
    
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.num_classes).to(device)

    
    for budget in budgets:

        results_per_budget[budget] = {}

        if hasattr(model, 'set_budget'):
            model.set_budget(budget)
            
        accs = []
        
        for val in noise_vals:

            if noise_module:
                noise_module.set_value(val)
            
            # compute accuracy given budget
            for batch, labels in tqdm(val_loader, desc=f'Testing epoch {epoch}, budget {budget}, noise: {noise_type} at {val}'):
                batch, labels = batch.to(device), labels.to(device)
                
                out = model(batch)
                predicted = torch.argmax(out, 1)
                metric.update(predicted, labels)
            
            acc = metric.compute()
            metric.reset()
            logger.log({f'test/budget_{budget}/noise_{val}': acc})
            accs.append(acc)

            flops = 0
            for batch, labels in tqdm(val_loader, desc=f'Counting flops for epoch {epoch} with budget {budget}'):
                batch, labels = batch.to(device), labels.to(device)
                num_flops, num_params = compute_flops(
                    model, 
                    batch,
                    as_strings=False,
                    verbose=False,
                    print_per_layer_stat=False,
                    flops_units='Mac'
                    )
                flops += num_flops
            # TODO check that there are no mistakes in the flops computation
            # if the flops are averaged over the batch 
            # check that the expected value is correct
            flops /= len(val_loader)
            
            if val is not None:
                results_per_budget[budget][val] = acc.item()
                results_per_flops[flops][val] = acc.item()
            else:
                results_per_budget[budget] = acc.item()
                results_per_flops[flops] = acc.item()

    logger.log({'flops': results_per_flops, 'budget': results_per_budget})
    
    return results_per_budget, results_per_flops




@hydra.main(version_base=None, config_path="../configs", config_name="test_config")
@torch.no_grad()
def test(cfg: DictConfig):

    # display config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(config_dict)

    # check arguments
    if cfg.load_from is None:
        raise ValueError('"load_from" must be specified to load a model from a checkpoint.')
    elif isinstance(cfg.load_from, str):
        load_from = [cfg.load_from]
    else:
        load_from = cfg.load_from

    # set seed and device
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # dataset and dataloader are the same for all tests
    dataset = instantiate(cfg.dataset)
    val_dataset = dataset.val_dataset
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.test.test_batch_size, 
        shuffle=False, 
        num_workers=cfg.test.num_workers, 
        pin_memory=True)

    
    # prepare dictionaries to store per-experiment results
    all_results_per_budget = {}
    all_results_per_flops = {}

    for experiment_dir in load_from:

        experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)
        logger = instantiate(cfg.logger, settings=str(config_dict), dir=experiment_dir)


        model_checkpoint = get_checkpoint_path(experiment_dir)
        print('Loading model from checkpoint: ', model_checkpoint)

        # validate
        results_per_budget, results_per_flops = validate(
            model_checkpoint, 
            logger,
            val_loader, 
            budgets=cfg.test.budgets,
            noise_settings=cfg.noise,
            noises=cfg.test.noises,
            device=device,
            )
        
        # these might be <buget -> noise -> acc > or <budget, acc>
        # in the first case we need to plot the results with noise
        noises = cfg.test.noises
        if noises is not None and len(noises) > 0:
            plot_budget_and_noise_recap(
                accs_per_budget=results_per_budget,
                accs_per_flops =results_per_flops,
                save_dir=os.path.join(experiment_dir, 'images'))
        else:
            plot_budget_recap(
                accs_per_budget=results_per_budget,
                accs_per_flops =results_per_flops,
                save_dir=os.path.join(experiment_dir, 'images'))        
        
        # store results in dictionary
        all_results_per_budget[experiment_dir] = results_per_budget
        all_results_per_flops[experiment_dir] = results_per_flops
    
    # plot cumulative results
    if cfg.test.cumulative_plot:
        
        cumulative_plot_dir = cfg.test.cumulative_plot_dir
        os.makedirs(cfg.test.cumulative_plot_dir, exist_ok=True)
        print('Saving cumulative plots to ', cumulative_plot_dir)
        

        noises = cfg.test.noises
        if noises is not None and len(noises) > 0:
            plot_cumulative_budget_and_noise_recap(
                all_results_per_flops, 
                additional_x_labels=cfg.test.budgets,
                save_dir=cumulative_plot_dir) 
        else:
            plot_cumulative_budget_recap(
                run_accs_per_budget=all_results_per_budget, 
                run_accs_per_flops=all_results_per_flops,
                save_dir=cumulative_plot_dir)
     

    
if __name__ == '__main__':
    test()