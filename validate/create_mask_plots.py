import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Any, List
from matplotlib import pyplot as plt
from collections import defaultdict


from omegaconf import DictConfig, OmegaConf
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
from hydra.utils import instantiate
from peekvit.utils.utils import (
    get_checkpoint_path,
    make_experiment_directory,
    load_state,
    add_noise,
)
import hydra
from pprint import pprint
from torch.utils.data import Subset


from peekvit.utils.utils import load_state, add_noise, make_experiment_directory
from peekvit.utils.visualize import (
    plot_budget_and_noise_recap,
    plot_cumulative_budget_recap,
    plot_budget_recap,
    plot_cumulative_budget_and_noise_recap,
)
from peekvit.utils.visualize import plot_masked_images


@hydra.main(
    version_base=None, config_path="../configs", config_name="test_config_personal"
)
@torch.no_grad()
def test(cfg: DictConfig):

    # display config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(config_dict)

    # check arguments
    if cfg.load_from is None:
        raise ValueError(
            '"load_from" must be specified to load a model from a checkpoint.'
        )
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
    subset_idcs = torch.arange(
        0, len(val_dataset), len(val_dataset) // cfg.test.num_images
    )
    images_to_plot = Subset(val_dataset, subset_idcs)
    budgets = cfg.test.budgets

    hard_mask = cfg.test.hard_mask
    hard_prefix = "hard_" if hard_mask else "soft_"  # just for plot names

    for experiment_dir in load_from:

        experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)

        # if model parameters are not specified in the config file, load the model from the checkpoint
        model_checkpoint = get_checkpoint_path(experiment_dir)
        print("Loading model from checkpoint: ", model_checkpoint)
        model, _, epoch, _, _ = load_state(model_checkpoint, model=None, strict=True)
        model.eval()
        model.to(device)

        # sanity check that model has budget
        if not hasattr(model, "set_budget"):
            print("Model does not have budget, setting to None")
            budgets = budgets or [1.0]

        if budgets is None or len(budgets) == 0:
            print("Budgets not specified, setting to None")
            budgets = [1.1]

        for budget in budgets:

            if hasattr(model, "set_budget"):
                model.set_budget(budget)

            images = plot_masked_images(
                model,
                images_to_plot,
                visualization_transform=dataset.denormalize_transform,
                hard=hard_mask,
            )

            os.makedirs(f"{experiment_dir}/images/epoch_{epoch}", exist_ok=True)
            os.makedirs(
                f"{experiment_dir}/images/epoch_{epoch}/budget_{budget}", exist_ok=True
            )
            for i, (_, img) in enumerate(images.items()):
                img.savefig(
                    f"{experiment_dir}/images/epoch_{epoch}/budget_{budget}/{hard_prefix}{subset_idcs[i]}.png"
                )


if __name__ == "__main__":
    test()
