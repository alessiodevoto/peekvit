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
from torch.utils.data import Subset


from peekvit.utils.utils import load_state, add_noise, make_experiment_directory
from peekvit.utils.visualize import plot_budget_and_noise_recap, plot_cumulative_budget_recap, plot_budget_recap, plot_cumulative_budget_and_noise_recap
from peekvit.utils.flops_count import compute_flops

from peekvit.utils.visualize import plot_class_tokens, plot_class_tokens_distances




def create_class_token_plots(
        model,
        images_to_plot: Subset,
        experiment_dir: str,
        model_checkpoint_path: str = None,
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        ):
    
    # if model parameters are not specified in the config file, load the model from the checkpoint
    if model_checkpoint_path is not None:
        model, _, epoch, _, _ = load_state(model_checkpoint_path, model=model, strict=True)
    
    model.eval()
    model.to(device)
    
    plots_dir = join(experiment_dir, 'cls_token_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    model.eval()

    for i, (image, label) in enumerate(images_to_plot):
        image = image.to(device)
        plot_class_tokens(model, image, savepath=join(plots_dir, f'cls_tokens_{i}.png'))
        plot_class_tokens_distances(model, image, savepath=join(plots_dir, f'cls_tokens_distances_{i}.png'))
    


@hydra.main(version_base=None, config_path="../configs", config_name="test_config")
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
    subset_idcs = torch.arange(0, len(val_dataset), len(val_dataset)//cfg.test.num_images)
    images_to_plot = Subset(val_dataset, subset_idcs)
    
    
    # if a model is provided in the config file, load it
    model = None
    if 'model' in cfg:
        print('Instantiating new model from config file. \nIf you want to load a model from a checkpoint, remove the "model" field from the config file.')
        model = instantiate(cfg.model)
        model = model.to(device)

    
    for experiment_dir in load_from:

        experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)
        logger = instantiate(cfg.logger, settings=str(config_dict), dir=experiment_dir)

        
        model_checkpoint_path = get_checkpoint_path(experiment_dir)
        print('Loading model from checkpoint: ', model_checkpoint_path)

        create_class_token_plots(
            model=model,
            model_checkpoint_path=model_checkpoint_path,
            images_to_plot=images_to_plot,
            experiment_dir=experiment_dir
        )
        
        
    
    
     

    
if __name__ == '__main__':
    test()