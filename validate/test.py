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
from hydra.utils import instantiate
from peekvit.utils.utils import get_checkpoint_path, make_experiment_directory, load_state, add_noise
import hydra
from pprint import pprint
import time

from peekvit.utils.utils import load_state, add_noise, make_experiment_directory
from peekvit.utils.visualize import plot_budget_and_noise_recap, plot_cumulative_budget_recap, plot_budget_recap, plot_timing_recap, plot_cumulative_budget_and_noise_recap
from peekvit.utils.flops_count import compute_flops


def move_dataset_to_device(dataloder, device):
    print(f'Moving dataset to device {device} for testing infernce speed')
    for batch, labels in dataloder:
        batch.to(device), labels.to(device)




def validate(
        model_checkpoint: str, 
        logger: Any,
        val_loader: DataLoader, 
        budgets: List,
        noise_settings: DictConfig,
        noises: List,
        device: torch.device,
        model: torch.nn.Module = None,
        ):
    

    # if model parameters are not specified in the config file, load the model from the checkpoint
    epoch = 'unknown'
    if model_checkpoint is not None:
        model, _, epoch, _, _ = load_state(model_checkpoint, model=model, strict=True)
    
    model.eval()
    model.to(device)

    move_dataset_to_device(val_loader, device)
    

    # sanity check that model has budget
    if not hasattr(model, 'set_budget'):
        print('Model does not have budget, setting to default value 1.1')
        budgets = [1.0]
    
    if budgets is None or len(budgets) == 0:
        print('Budgets not specified, setting to default value 1.0')
        budgets = [1.0]
    
    if hasattr(model, 'enable_ranking'):
        print('Detected model with ranking capabilities. Enabling ranking for testing.')
        model.enable_ranking(True)
        
    
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
    
    timings_per_budget = defaultdict(dict)
    timings_per_flops = defaultdict(dict)

    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.num_classes).to(device)

    
    for budget in budgets:

        results_per_budget[budget] = {}
        timings_per_budget[budget] = {}

        if hasattr(model, 'set_budget'):
            model.set_budget(budget)
            
        accs = []
        
        for noise_val in noise_vals:

            if noise_module:
                noise_module.set_value(noise_val)
            
            start_time = time.time()
            
            # compute accuracy given budget and noise
            for batch, labels in tqdm(val_loader, desc=f'Testing epoch {epoch}, budget {budget}, noise: {noise_type} - {noise_val}'):
                batch, labels = batch.to(device), labels.to(device)
                
                out = model(batch)
                predicted = torch.argmax(out, 1)
                metric.update(predicted, labels)

            elapsed_time = time.time() - start_time
            images_per_second = len(val_loader.dataset) / elapsed_time
            
            
            acc = metric.compute()
            metric.reset()
            logger.log({f'test/budget_{budget}/noise_{noise_val}': acc})
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
            
            if noise_val is not None:
                results_per_budget[budget][noise_val] = acc.item()
                results_per_flops[flops][noise_val] = acc.item()
                timings_per_budget[budget][noise_val] = images_per_second
                timings_per_flops[flops][noise_val] = images_per_second
            else:
                results_per_budget[budget] = acc.item()
                results_per_flops[flops] = acc.item()
                timings_per_budget[budget] = images_per_second
                timings_per_flops[flops] = images_per_second

    logger.log({'flops': results_per_flops, 'budget': results_per_budget, 'timings_flops': timings_per_flops, 'timings_budget': timings_per_budget})
    # print('Results per budget: ', results_per_budget)
    # print('Results per flops: ', results_per_flops)
    return results_per_budget, results_per_flops, timings_per_budget, timings_per_flops




@hydra.main(version_base=None, config_path="../configs", config_name="test_config_personal")
@torch.no_grad()
def test(cfg: DictConfig):


    # display config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(config_dict)

    # set seed and device
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # check arguments
    if cfg.load_from is None:
        print('No model checkpoint provided.')
        l, _ = make_experiment_directory(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        load_from = [l]
    elif isinstance(cfg.load_from, str):
        load_from = [cfg.load_from]
    else:
        load_from = cfg.load_from
    


    # dataset and dataloader are the same for all tests
    dataset = instantiate(cfg.dataset)
    val_dataset = dataset.val_dataset
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.test.test_batch_size, 
        shuffle=False, 
        num_workers=cfg.test.num_workers, 
        pin_memory=True)
    
    # if a model is provided in the config file, load it
    model = None
    if 'model' in cfg:
        print('Instantiating model from config file.')
        model = instantiate(cfg.model)
        model = model.to(device)

    
    # prepare dictionaries to store per-experiment results
    all_results_per_budget = {}
    all_results_per_flops = {}

    for experiment_dir in load_from:

        experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)
        logger = instantiate(cfg.logger, settings=str(config_dict), dir=experiment_dir)


        model_checkpoint_path = get_checkpoint_path(experiment_dir)
        if model_checkpoint_path is not None:
            print('Loading model from checkpoint: ', model_checkpoint_path)
        else:
            print('No model checkpoint found in ', experiment_dir)
            print('If you are trying to load the model from a local checkpoint, please check the path.')
            print('If you are loading the model from a config file, ignore this message.')
            

        # validate
        results_per_budget, results_per_flops, timings_per_budgets, timings_per_flops = validate(
            model_checkpoint_path, 
            logger,
            val_loader, 
            budgets=cfg.test.budgets,
            noise_settings=cfg.noise,
            noises=cfg.test.noises,
            device=device,
            model=model,
            )
        
        # these might be {flops/budget : {noise : acc} or {flops/budget : acc}
        # in the first case we need to plot the results with noise
        noises = cfg.test.noises
        validating_with_noise = noises is not None and len(noises) > 0 and cfg.noise != {}

        if validating_with_noise:
            plot_budget_and_noise_recap(
                accs_per_budget=results_per_budget,
                accs_per_flops =results_per_flops,
                save_dir=os.path.join(experiment_dir, 'images'))
        else:
            plot_budget_recap(
                accs_per_budget=results_per_budget,
                accs_per_flops =results_per_flops,
                save_dir=os.path.join(experiment_dir, 'images'))  
            
            plot_timing_recap(
                timings_per_budgets,
                timings_per_flops,
                save_dir=os.path.join(experiment_dir, 'images'))


        
        # store results in dictionary
        all_results_per_budget[experiment_dir] = results_per_budget
        all_results_per_flops[experiment_dir] = results_per_flops

    # Notice that all_results_per_flops is a dictionary of dictionaries
    # If validating with noise, it is like this: 
        # {experiment_dir : {flops : {noise : acc}}}
    # If not validating with noise, it is like this:
        # {experiment_dir : {flops : acc}}
    
    # plot cumulative results in case we have more than one experiment
    if cfg.test.cumulative_plot:
        
        cumulative_plot_dir = cfg.test.cumulative_plot_dir
        os.makedirs(cfg.test.cumulative_plot_dir, exist_ok=True)
        print('Saving cumulative plots to ', cumulative_plot_dir)
        

        # noises = cfg.test.noises
        # validating_with_noise = noises is not None and len(noises) > 0 and cfg.noise != {}
        if validating_with_noise:
            plot_cumulative_budget_and_noise_recap(
                all_results_per_flops, 
                additional_x_labels=cfg.test.budgets,
                save_dir=cumulative_plot_dir,
                run_names=cfg.test.run_names
                ) 
        else:
            plot_cumulative_budget_recap(
                run_accs_per_budget=all_results_per_budget, 
                run_accs_per_flops=all_results_per_flops,
                save_dir=cumulative_plot_dir,
                run_names=cfg.test.run_names
                )
     

    
if __name__ == '__main__':
    test()