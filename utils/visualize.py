from collections import defaultdict
import os
import re
from matplotlib import cm
import torch
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from typing import List, Optional, Tuple
from tqdm import tqdm
from plotly import graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from pathlib import Path
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


import wandb

from .utils import make_batch, get_model_device, get_last_forward_gates, get_moes, get_forward_masks, get_learned_thresholds

def string_to_color(s):
    """
    Convert a string to a unique color using its hash value.
    """
    hash_value = hash(str(s))
    color = plt.cm.tab10(hash_value % 10)   # Use viridis colormap, but you can choose any colormap
    return color

######################################################## Utils ##################################################################

def prepare_for_matplotlib(t):
    """
    Prepares the given tensor for matplotlib by converting it to a numpy array and reshaping it to have channels last.
    If it is a pytorch tensor, converts it to a numpy array and reshapes it to have channels last.
    If it is a numpy a numpy array, reshapes it to have channels last.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if isinstance(t, np.ndarray) and len(t.shape) == 3 and t.shape[0] in {3, 1}:
        t = rearrange(t, 'c h w -> h w c')
        
    return t


def denormalize(t: torch.Tensor, mean: Tuple, std: Tuple):
    """
    Denormalizes the given tensor with the given mean and standard deviation.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return t * std + mean


######################################################## Common ##################################################################

def plot_budget_recap(accs_per_budget, accs_per_flops, save_dir, additional_label=""):
    """
    Plots the budget recap by plotting the accuracy values against the budget values.
    
    Parameters:
    - accs_per_budget (dict): A dictionary mapping budget values to accuracy values.
    - accs_per_flops (dict): A dictionary mapping budget values to accuracy values.
    - save_dir (str): The directory where the plot images will be saved.
    """
    if accs_per_budget is not None:
      fig, ax = plt.subplots()
      ax.plot(accs_per_budget.keys(), accs_per_budget.values(), marker='o')
      ax.set_xlabel('Budget')
      ax.set_ylabel('Accuracy')
      ax.set_title('Budget vs Accuracy')
      plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      plt.ylim([0.1, 1.0])
      plt.savefig(os.path.join(save_dir, f'budget_vs_acc{additional_label}.png'))

    if accs_per_flops is not None:
      fig, ax = plt.subplots()
      ax.plot(accs_per_flops.keys(), accs_per_flops.values(), marker='o')
      ax.set_xlabel('Flops')
      ax.set_ylabel('Accuracy')
      ax.set_title('Flops vs Accuracy')
      plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      plt.ylim([0.1, 1.0])
      plt.savefig(os.path.join(save_dir, f'flops_vs_acc{additional_label}.png'))


def plot_timing_recap(timings_per_budgets, timings_per_flops, save_dir, additional_label=""):
    if timings_per_budgets is not None:
      fig, ax = plt.subplots()
      ax.plot(timings_per_budgets.keys(), timings_per_budgets.values(), marker='o')
      ax.set_xlabel('Budget')
      ax.set_ylabel('Throughput (images/s)')
      ax.set_title('Budget vs Throughput')
      plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      plt.savefig(os.path.join(save_dir, f'budget_vs_throughput{additional_label}.png'))

    if timings_per_flops is not None:
      fig, ax = plt.subplots()
      ax.plot(timings_per_flops.keys(), timings_per_flops.values(), marker='o')
      ax.set_xlabel('Flops')
      ax.set_ylabel('Throughput (images/s)')
      ax.set_title('Flops vs Throughput')
      plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      plt.savefig(os.path.join(save_dir, f'flops_vs_throughput{additional_label}.png'))


def plot_cumulative_budget_recap(run_accs_per_budget, run_accs_per_flops, save_dir, additional_label="", run_names=None):
    if run_accs_per_budget is not None:
      fig, ax = plt.subplots()
      for i , (run_id, accs_per_budget) in enumerate(run_accs_per_budget.items()):
        ax.plot(accs_per_budget.keys(), accs_per_budget.values(), marker='o', color=string_to_color(i))
        ax.set_xlabel('Budget')
        ax.set_ylabel('Accuracy')
        ax.set_title('Budget vs Accuracy')
        plt.ylim([0.1, 1.0])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
      plt.legend(run_names or [x.split('/')[-1] for x in run_accs_per_flops.keys()])
      plt.savefig(os.path.join(save_dir, f'cumulative_budget_vs_acc{additional_label}.png'))
    
    if run_accs_per_flops is not None:
      fig, ax = plt.subplots()
      for i, (run_id, accs_per_flops) in enumerate(run_accs_per_flops.items()):
        ax.plot(accs_per_flops.keys(), accs_per_flops.values(), marker='o', color=string_to_color(i))
        ax.set_xlabel('Flops')
        ax.set_ylabel('Accuracy')
        ax.set_title('Flops vs Accuracy')
        plt.ylim([0.1, 1.0])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      
      plt.legend(run_names or [x.split('/')[-1] for x in run_accs_per_flops.keys()])
      plt.savefig(os.path.join(save_dir, f'cumulative_flops_vs_acc{additional_label}.png'))


def plot_budget_and_noise_recap(accs_per_budget, accs_per_flops, save_dir, additional_label=""):
    if accs_per_budget is not None:
      fig, ax = plt.subplots()
      for budget, results in accs_per_budget.items():
          ax.plot(results.keys(), results.values(), marker='o', label=f'budget {budget}')
          ax.set_xlabel('Noise')
          ax.set_ylabel('Accuracy')
          ax.set_title('Noise vs Accuracy across budgets')
          ax.legend()

      plt.ylim([0.1, 1.0])
      plt.savefig(os.path.join(save_dir, f'budget_vs_noise_vs_acc{additional_label}.png'))

      print(accs_per_budget)
      fig, ax = plt.subplots()
      # flip the dictionary so that we have noise as the first key
      results_per_noise = {}
      for budget, results in accs_per_budget.items():
          for noise, acc in results.items():
              if noise not in results_per_noise:
                  results_per_noise[noise] = {}
              results_per_noise[noise][budget] = acc
      print(results_per_noise)
      # plot the noise vs accuracy
      for noise, results in results_per_noise.items():
          ax.plot(results.keys(), results.values(), marker='o', label=f'noise {noise}')
          ax.set_xlabel('Budget')
          ax.set_ylabel('Accuracy')
          ax.set_title('Budget vs Accuracy across noises')
          ax.legend()
      plt.ylim([0.1, 1.0])
      plt.savefig(os.path.join(save_dir, f'noise_vs_budget_vs_acc{additional_label}.png'))


    if accs_per_flops is not None:
      fig, ax = plt.subplots()
      for budget, results in accs_per_flops.items():
          ax.plot(results.keys(), results.values(), marker='o', label=f'budget {budget}')
          ax.set_xlabel('Noise')
          ax.set_ylabel('Accuracy')
          ax.set_title('Noise vs Accuracy across flops')
          ax.legend()
      # plt.ylim([0.4, 0.9])
      plt.savefig(os.path.join(save_dir, f'flops_vs_noise_vs_acc{additional_label}.png'))
    
    
def plot_cumulative_budget_and_noise_recap(run_accs_per_flops, save_dir, additional_x_labels="", run_names=None):

    results_per_noise = {}
    for exp_dir, flops_data in run_accs_per_flops.items():
        for flop, noise_data in flops_data.items():
            for noise, acc in noise_data.items():
                if noise not in results_per_noise:
                    results_per_noise[noise] = {}
                if exp_dir not in results_per_noise[noise]:
                    results_per_noise[noise][exp_dir] = {}
                results_per_noise[noise][exp_dir][flop] = acc

    print(results_per_noise)

    for noise, exps in results_per_noise.items():
        plot_cumulative_budget_recap(run_accs_per_budget=None, run_accs_per_flops=exps, save_dir=save_dir, additional_label=f'_noise_{noise}', run_names=run_names) 
       

def plot_cumulative_budget_and_noise_recap_old(run_accs_per_flops, save_dir, additional_x_labels=""):
    # results per budget is a dict of dicts, where the first key is the budget and 
    # the second key is the noise value, and the value is the accuracy.
    # we want to create a plot where the x axis is the budget, the y axis is the accuracy, 
    # and we have a line for each noise

    # create new dictionary with noise as first key and budget as second key


    _results_per_noise = {}
    for run in run_accs_per_flops:
        _results_per_noise[run] = defaultdict(dict)
        for budget in run_accs_per_flops[run]:   
            for noise in run_accs_per_flops[run][budget]:
                _results_per_noise[run][noise][budget if budget not in {0.0, float('inf')} else 1.1] = run_accs_per_flops[run][budget][noise]

    print(_results_per_noise)


    fig, ax = plt.subplots()
    
    for run, results_per_noise in _results_per_noise.items():
        is_base_run = 'base' in run
        for noise, results in results_per_noise.items():
            ax.plot(results.keys(), results.values(), marker='o' if not is_base_run else '*', label=f'noise {noise}', color=string_to_color(str(noise)))
            ax.set_xlabel(f'Budgets {additional_x_labels if additional_x_labels is not None else ""}')
            ax.set_ylabel('Accuracy')
            ax.set_title('Budget vs Accuracy across Noises')
            # set y range
            # # plt.ylim([0.1, 1.0])
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid(visible=True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    
    
    # create save dir if it does not exist
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'cumulative_budget_vs_acc_noises.png'))
       
      
    return fig


######################################################## OLD ##################################################################
def plot_budget_vs_acc(budgets, accs, epoch, save_dir):
  """
  Plots the accuracy vs budget curve for the given budgets and accuracies.

  Args:
    budgets (List): The budgets.
    accs (List): The accuracies.
    save_dir (str): The directory to save the plot.

  Returns:
    None
  """
  fig, ax = plt.subplots()
  ax.plot(budgets, accs, marker='o')

  # set labels
  ax.set_xlabel('Budget')
  ax.set_ylabel('Accuracy')
  ax.set_title('Budget vs Accuracy')

  # set y range
  # # plt.ylim([0.4, 0.9])
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

  # create save dir if it does not exist
  if save_dir is not None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'budget_vs_acc_{epoch}.png'))
  
  return fig


def plot_budget_vs_sparsity(budgets, accs, epoch, save_dir):
  """
  Plots the accuracy vs budget curve for the given budgets and accuracies.

  Args:
    budgets (List): The budgets.
    accs (List): The accuracies.
    save_dir (str): The directory to save the plot.

  Returns:
    None
  """
  fig, ax = plt.subplots()
  ax.plot(budgets, accs, marker='o')

  # set labels
  ax.set_xlabel('Budget')
  ax.set_ylabel('Sparsity')
  ax.set_title('Budget vs Sparsity')

  # set y range
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

  # create save dir if it does not exist
  if save_dir is not None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'budget_vs_sparsity_{epoch}.png'))
  
  return fig


def plot_noise_vs_acc(budgets, accs, epoch, save_dir, additional_label=None):
  """
  Plots the accuracy vs noise curve for the given noises and accuracies.

  Args:
    noise (List): The noises.
    accs (List): The accuracies.
    save_dir (str): The directory to save the plot.

  Returns:
    None
  """
  fig, ax = plt.subplots()
  ax.plot(budgets, accs, marker='o')

  # set labels
  ax.set_xlabel('Noise')
  ax.set_ylabel('Accuracy')
  ax.set_title('Noise vs Accuracy' if additional_label is None else f'Noise vs Accuracy ({additional_label})')

  # set y range
  # # plt.ylim([0.1, 1.0])

  # create save dir if it does not exist
  if save_dir is not None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'noise_vs_acc_{epoch}.png'))
  
  return fig


def plot_budget_vs_noise_vs_acc(results_per_budget: dict,  save_dir: str = None):
    # results per budget is a dict of dicts, where the first key is the budget and 
    # the second key is the noise value, and the value is the accuracy.
    # we want to create a plot where the x axis is the noise value, the y axis is the accuracy, 
    # and we have a line for each budget

    fig, ax = plt.subplots()
    
    for budget, results in results_per_budget.items():
        ax.plot(results.keys(), results.values(), marker='o', label=f'budget {budget}')
        ax.set_xlabel('Noise')
        ax.set_ylabel('Accuracy')
        ax.set_title('Noise vs Accuracy across budgets')
        ax.legend()


        # create save dir if it does not exist
        if save_dir is not None:
          Path(save_dir).mkdir(parents=True, exist_ok=True)
          plt.savefig(os.path.join(save_dir, f'noise_vs_acc_budgets.png'))
    
    return fig
  

def plot_model_budget_vs_noise_vs_acc(results_per_model: dict, save_dir: str = None):
    # results per budget is a dict of dicts, where the first key is the budget and 
    # the second key is the noise value, and the value is the accuracy.
    # we want to create a plot where the x axis is the noise value, the y axis is the accuracy, 
    # and we have a line for each budget

    fig, ax = plt.subplots()

    for model_name, results_per_budget in results_per_model.items():
    
      for budget, results in results_per_budget.items():
          ax.plot(results.keys(), results.values(), marker='o', label=f'budget {budget}')
          ax.set_xlabel('Noise')
          ax.set_ylabel('Accuracy')
          ax.set_title('Noise vs Accuracy across budgets')
          ax.legend()
          # set y range
          # # plt.ylim([0.1, 1.0])

          # create save dir if it does not exist
          if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'noise_vs_acc_budgets.png'))
      
    return fig


def plot_model_noise_vs_budget_vs_acc(results_per_model: dict, save_dir: str = None, additional_x_labels: List = None):
    # results per budget is a dict of dicts, where the first key is the budget and 
    # the second key is the noise value, and the value is the accuracy.
    # we want to create a plot where the x axis is the budget, the y axis is the accuracy, 
    # and we have a line for each noise

    # create new dictionary with noise as first key and budget as second key

    _results_per_noise = {}
    for run in results_per_model:
        _results_per_noise[run] = defaultdict(dict)
        for budget in results_per_model[run]:   
            for noise in results_per_model[run][budget]:
                _results_per_noise[run][noise][budget if budget not in {0.0, float('inf')} else 1.1] = results_per_model[run][budget][noise]

    print(_results_per_noise)


    fig, ax = plt.subplots()
    
    for run, results_per_noise in _results_per_noise.items():
        is_base_run = 'base' in run
        for noise, results in results_per_noise.items():
            ax.plot(results.keys(), results.values(), marker='o' if not is_base_run else '*', label=f'noise {noise}', color=string_to_color(str(noise)))
            ax.set_xlabel(f'Budgets {additional_x_labels if additional_x_labels is not None else ""}')
            ax.set_ylabel('Accuracy')
            ax.set_title('Budget vs Accuracy across Noises')
            # set y range
            # # plt.ylim([0.1, 1.0])

            
            #x_ticks = [f'{x}({y})' for x,y in zip(results.keys(), additional_x_labels)] if additional_x_labels is not None else results.keys()
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid(visible=True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    
    
    # create save dir if it does not exist
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'budget_vs_acc_noises.png'))
       
      
    return fig


######################################################## MoEs ##################################################################


@torch.no_grad()
def img_expert_distribution(model, images: List, subset, transform: Optional[None] = None, save_dir: str = None):
  """
  Visualizes the expert distribution for each MOE layer in the model.
  If save_dir is not None, each moe layer output will be saved in a new subfolder of save_dir named after the moe layer.

  Args:
    model (torch.nn.Module): The MOE model.
    images (List): List of input images.
    subset: The indices of images in `images` to visualize.
    transform (Optional[None]): Optional image transformation function, to be applied to each image before forwarding to model. Default is None.
    save_dir (str): Directory to save the visualizations. Default is None.

  Returns:
    None
  """

  model.eval()
  device = get_model_device(model)

  image_size = max(images[0][0].shape[-1], images[0][0].shape[0])  # it could be channel first or channel last
  patch_size = model.patch_size
  patches_per_side = (image_size // patch_size)
  num_registers = getattr(model, 'num_registers', 0)
  num_class_tokens = getattr(model, 'num_class_tokens', 1)

  for img_idx in tqdm(subset, desc='Preparing expert distribution plots'):

    img, label = images[img_idx]
    # forward pass
    _img = transform(img) if transform is not None else img
    model(make_batch(_img).to(device))

    # retrieve last forward gating probs
    gates = get_last_forward_gates(model)  # <moe, gating_probs>, each gating_probs is (batch, tokens, exp)

    # prepare plot, we want a row for each moe layer,
    # and two columns, one for the image and one for the expert distribution
    fig, axs = plt.subplots(len(gates.keys()), 2, squeeze=False)

    # for each moe layer, plot the distribution of experts
    for moe_idx, (moe_name, gating_probs) in enumerate(gates.items()):

      # we assume top-1 gating
      # gating probs is (batch, tokens, exp)
      max_vals, max_idx = gating_probs.max(dim=-1)
      exp_assignment = max_idx[:, num_class_tokens + num_registers:].reshape(-1, patches_per_side, patches_per_side)  # discard class token and reshape as image
      img, exp_assignment = prepare_for_matplotlib(img), prepare_for_matplotlib(exp_assignment)
      axs[moe_idx, 0].imshow(img)
      axs[moe_idx, 1].imshow(exp_assignment)

      axs[moe_idx, 0].title.set_text(moe_name)

    fig.tight_layout()

    if save_dir is not None:
      os.makedirs(save_dir, exist_ok=True)
      plt.savefig(join(save_dir, f'expert_distribution_batch_{img_idx}.jpg'), dpi=100)
    plt.close()


def display_expert_embeddings(model, save_dir):
  """
  Display the expert embeddings using a 3D scatter plot. We use PCA to reduce dimensionality of expert embeddings.

  Args:
    model (Model): The model.
    save_dir (str): The directory to save the generated plots.

  Returns:
    None
  """
  moes = get_moes(model)
  for moe_name, moe in moes.items():
    
    # get the expert embeddings, shape (num_experts, hidden_dim)
    embs = moe.gating_network.gate.weight.detach().cpu().numpy()

    # reduce hidden_dim to 3 using PCA
    pca = PCA(n_components=3)
    transformed_data = pca.fit_transform(embs)
    fig = go.Figure(go.Scatter3d(x=transformed_data[:, 0],
                   y=transformed_data[:, 1],
                   z=transformed_data[:, 2],
                   mode='markers',
                   marker=dict(size=12,
                         color=list(range(embs.shape[0])),
                         opacity=0.8)))
    fig.update_layout(title=moe_name)

    # save
    if save_dir is not None:
      os.makedirs(save_dir, exist_ok=True)
      fig.write_image(join(save_dir, f'{moe_name}_experts.png'))


######################################################## Residual ##################################################################

@torch.no_grad()
def plot_masked_images(
   model, 
   images, 
   model_transform=None, 
   visualization_transform=None, 
   hard=True,
   skip_layers: List[int] = [],
   overlay: bool = False,
   ):
  model.eval()
  device = get_model_device(model)
  num_registers = getattr(model, 'num_registers', 0) 
  num_class_tokens = getattr(model, 'num_class_tokens', 1)
  num_budget_tokens = getattr(model, 'num_budget_tokens', 0)

  
  image_size = max(images[0][0].shape[-1], images[0][0].shape[0])  # it could be channel first or channel last
  patch_size = model.patch_size
  patches_per_side = (image_size // patch_size)
  
  figs = {}
  i = 0
  for img, label in tqdm(images, desc='Preparing masked images plots'):
    
    # forward pass
    _img = model_transform(img) if model_transform is not None else img
    
    # model.set_budget(budget)
    out = model(make_batch(_img).to(device)) 

    gates = get_forward_masks(model, incremental=True) 

    # prepare plot, we want a row for each residual layer,
    # and two columns, one for the image and one for token masks
    fig, axs = plt.subplots(len(gates.keys())+1 - len(skip_layers), 1, squeeze=False, figsize=(3, 25))

    # plot the image
    img = prepare_for_matplotlib(visualization_transform(img) if visualization_transform is not None else img)
    axs[0,0].imshow(img)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])

    # for each layer, plot the image and the token mask
    plot_idx = 1
    for layer_idx, (layer_name, forward_mask) in enumerate(gates.items()):

      if layer_idx in skip_layers:
        continue
      
      forward_mask = forward_mask[:, num_class_tokens+num_registers-1:].detach().reshape(-1, patches_per_side, patches_per_side)  # discard class token and reshape as image
      
      # replace non-zero values with 1
      if hard:
        forward_mask = forward_mask.ceil()
      
      if overlay:
        axs[plot_idx,0].imshow(img)
        
      forward_mask = prepare_for_matplotlib(forward_mask)
      im = axs[plot_idx,0].imshow(forward_mask, vmin=0, vmax=1, alpha=0.1 if overlay else 1, cmap='Reds' if overlay else 'viridis')

      
      
      # remove left and right margin 
      plt.tight_layout()

      # remove ticks on x and y axis
      axs[plot_idx,0].set_xticks([])
      axs[plot_idx,0].set_yticks([])

      plot_idx += 1

      #axs[layer_idx+1,0].title.set_text(layer_name) # to display transf layer name
      #cbar = axs[layer_idx+1,0].figure.colorbar(im, ax=axs[layer_idx+1,0], orientation='horizontal', shrink=0.2) # to display colorbar

      # set title to predicted and ground truth class
      try:
        # title = f'Predicted class: {IMAGENETTE_CLASSES[torch.argmax(out).item()]} Ground truth class: {IMAGENETTE_CLASSES[label]}'
        #axs[0,0].title.set_text(title)
         pass
      except Exception as e:
        pass

    fig.tight_layout()

    figs[f'mask_{i}'] = fig
    i+= 1

  plt.close()
  
  return figs

  

@torch.no_grad()
def img_mask_distribution(
  model, 
  images: List, 
  subset, 
  model_transform: Optional[None] = None, 
  visualization_transform: Optional[None] = None, 
  save_dir: str = None, 
  hard: bool = False, 
  budget: str = None,
  log_to_wandb: bool = False):
  """
  Visualizes the masking distribution of an image using a given model.

  Args:
    model (nn.Module): The model used for masking.
    images (List): A list of images to visualize.
    subset: The subset of images to visualize.
    model_transform (Optional[None]): An optional transformation to apply to the images before passing them through the model. Default is None.
    visualization_transform (Optional[None]): An optional transformation to apply to the images before visualization. Default is None.
    save_dir (str): The directory to save the visualization images. Default is None.
    hard (bool): Whether to use hard masking (1s and 0s) instead of soft masking. Default is False.
    budget (str): The budget to use for the visualization. Default is None.
    log_to_wandb (bool): Whether to log the images to wandb. Default is False.
  """
  
  model.eval()
  device = get_model_device(model)
  num_registers = getattr(model, 'num_registers', 0) 
  num_class_tokens = getattr(model, 'num_class_tokens', 1)
  num_budget_tokens = getattr(model, 'num_budget_tokens', 0)
  
  image_size = max(images[0][0].shape[-1], images[0][0].shape[0])  # it could be channel first or channel last
  patch_size = model.patch_size
  patches_per_side = (image_size // patch_size)

  figs = {}
  for img_idx in tqdm(subset, desc='Preparing masking plots'):

    img, label = images[img_idx]
    # forward pass
    _img = model_transform(img) if model_transform is not None else img
    # model.set_budget(budget)
    out = model(make_batch(_img).to(device))
    
    from peekvit.data.imagenette import IMAGENETTE_CLASSES
    #print(f'Predicted class: {IMAGENETTE_CLASSES[torch.argmax(out).item()]} Ground truth class: {IMAGENETTE_CLASSES[label]}')

    # retrieve last forward masks
    gates = get_forward_masks(model, incremental=True)  # <residual, gating_probs>
    thresholds = get_learned_thresholds(model) # <residual, threshold>
    
    # prepare plot, we want a row for each residual layer,
    # and two columns, one for the image and one for token masks
    fig, axs = plt.subplots(len(gates.keys())+1, 1, squeeze=False, figsize=(10, 25))

    # plot the image
    img = prepare_for_matplotlib(visualization_transform(img) if visualization_transform is not None else img)
    axs[0,0].imshow(img)

    # for each layer, plot the image and the token mask
    for layer_idx, (layer_name, forward_mask) in enumerate(gates.items()):
      
      forward_mask = forward_mask[:, num_class_tokens+num_registers-1:].detach().reshape(-1, patches_per_side, patches_per_side)  # discard class token and reshape as image
      
      # replace non-zero values with 1
      if hard:
        # forward_mask = torch.nn.functional.relu(forward_mask - thresolds[layer_name]).ceil()
        # print(torch.any(forward_mask >= 0.5))
        # forward_mask = forward_mask.round()
        forward_mask = forward_mask.ceil()
        
      forward_mask = prepare_for_matplotlib(forward_mask)
      im = axs[layer_idx+1,0].imshow(forward_mask, vmin=0, vmax=1)
      axs[layer_idx+1,0].title.set_text(layer_name)
      cbar = axs[layer_idx+1,0].figure.colorbar(im, ax=axs[layer_idx+1,0], orientation='horizontal', shrink=0.2)

      # set title to predicted and ground truth class
      try:
        title = f'Predicted class: {IMAGENETTE_CLASSES[torch.argmax(out).item()]} Ground truth class: {IMAGENETTE_CLASSES[label]}'
        axs[0,0].title.set_text(title)
      except Exception as e:
        pass

    fig.tight_layout()

    if save_dir is not None:
      os.makedirs(save_dir, exist_ok=True)
      plt.tight_layout()
      plt.savefig(join(save_dir, f'token_masks_batch_{img_idx}.jpg'), dpi=100)
    
    if log_to_wandb:
      #wandb.log({f'{budget}/token_masks_batch_{img_idx}.jpg': wandb.Image(plt)})
      figs[f'images/{budget}/token_masks_batch_{img_idx}.jpg'] = fig

  if log_to_wandb:
    wandb.log(figs)

  plt.close()


######################################################## Common ##################################################################

@torch.no_grad()
def get_cls_tokens(model, input):
    """
    Retrieves the class tokens from the model's output.

    Args:
      model (torch.nn.Module): The model to extract class tokens from.
      input (torch.Tensor): The input tensor to the model.

    Returns:
      dict: A dictionary mapping exit module name to its class token output.
    """
    
    num_class_tokens = getattr(model, 'num_class_tokens', 1)
    if num_class_tokens > 1:
      raise NotImplementedError('Only one class token is supported at the moment.')
    
    # get all layers with a regex
    all_layers = " ".join(get_graph_node_names(model)[0])
    pattern = re.compile(r'encoder\.layers\.\d+')
    matches = list(set(pattern.findall(all_layers)))

    # dictioanry mapping layer name to personal exit name
    # see https://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html?highlight=create_feature_extractor#torchvision.models.feature_extraction.create_feature_extractor
    exit_modules = {layer: f'layer_{i}' for i, layer in enumerate(matches)}

    model = create_feature_extractor(model, exit_modules)
    out = model(make_batch(input))

    # out is a dictionary mapping exit module name to its output
    # the class token is the first token of the output
    out = {key: value[:, :1] for key, value in out.items()}

    return out


@torch.no_grad()
def plot_class_tokens(model, input, save_dir=None, savepath=None):
    """
    Plots the class tokens of a given model and input.

    Args:
        model (torch.nn.Module): The model to visualize.
        input (torch.Tensor): The input to the model.
        save_dir (str, optional): The directory to save the plot. Defaults to None.
        savepath (str, optional): The path to save the plot. Defaults to None.
    """

    # xor save_dir and savepath
    assert (save_dir is None) != (savepath is None), 'Either save_dir or savepath must be specified, but not both.'

    cls_tokens = get_cls_tokens(model, input)

    all_exits = torch.stack(list(cls_tokens.values()))
    data_np = all_exits.squeeze().t().cpu().numpy()
    
    plt.imshow(data_np, cmap='viridis', aspect='auto')
    plt.xlabel('transformer layer')
    plt.ylabel('dimension')

    # add vertical lines to separate transformer layers
    for i in range(1, len(cls_tokens)):
      plt.axvline(x=i - 0.5, color='white', linewidth=2)
    

    if save_dir is not None:
      os.makedirs(save_dir, exist_ok=True)
      plt.savefig(join(save_dir, f'class_tokens.jpg'), dpi=200)
    elif savepath is not None:
      plt.savefig(savepath, dpi=100)
    
    plt.close()

    

@torch.no_grad()
def plot_class_tokens_distances(model, input, save_dir=None, savepath=None):
    
    # xor save_dir and savepath
    assert (save_dir is None) != (savepath is None), 'Either save_dir or savepath must be specified, but not both.'
   
    cls_tokens = get_cls_tokens(model, input)
    all_exits = torch.stack(list(cls_tokens.values()))

    # we compute the distance between each pair of class tokens and display it as a heatmap
    distances = torch.cdist(all_exits.squeeze(), all_exits.squeeze())

    plt.imshow(distances.cpu().numpy(), cmap='viridis', aspect='auto')

    # adjust labels and move xtickts to top
    plt.xticks(np.arange(len(cls_tokens)), sorted(list(cls_tokens.keys())))
    plt.yticks(np.arange(len(cls_tokens)), sorted(list(cls_tokens.keys())))
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.colorbar()


    if save_dir is not None:
      os.makedirs(save_dir, exist_ok=True)
      plt.savefig(join(save_dir, f'class_tokens_distances.jpg'), dpi=100)
    elif savepath is not None:
      plt.savefig(savepath, dpi=100)
    
    plt.close()
    
    

def plot_reconstructed_images(model, images_to_plot, model_transform, visualization_transform):
  model.eval()
  device = get_model_device(model)
  figs = {}
  
  i = 0
  for img, label in tqdm(images_to_plot, desc='Preparing reconstructed images plots'):
    
    # forward pass
    _img = model_transform(img) if model_transform is not None else img
    
    # model.set_budget(budget)
    out, reconstructed, mask = model(make_batch(_img).to(device)) 
    


    # prepare plot, we want a row for each residual layer,
    # and two columns, one for the image and one for token masks
    fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(10, 25))

    # plot the image
    img = prepare_for_matplotlib(visualization_transform(img) if visualization_transform is not None else img)
    axs[0,0].imshow(img)
    axs[0,0].title.set_text('Original image')

    # plot the reconstructed image
    reconstructed = prepare_for_matplotlib(visualization_transform(reconstructed * (1-mask)).squeeze())
    axs[1,0].imshow(reconstructed)
    axs[1,0].title.set_text('Reconstructed image')

    fig.tight_layout()

    figs[f'reconstructed_{i}'] = fig
    i+=1

  
  return figs
  
  
 

  
  

######################################################## Example usage ##################################################################

"""image_path = "no_brain.jpg"
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
resized_image = transform(image)

expert_distribution(vitmoe, [torch.stack([resized_image, resized_image], dim=0)], image_size=64)

token_distribution(model, [torch.stack([resized_image, resized_image], dim=0)], image_size=64)
"""


