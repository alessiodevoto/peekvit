import os
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

from utils.utils import make_batch, get_model_device, get_last_forward_gates, get_moes, get_forward_masks, get_learned_thresholds

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
def img_mask_distribution(model, images: List, subset, model_transform: Optional[None] = None, visualization_transform: Optional[None] = None, save_dir: str = None, hard: bool = False):
  """
  Plot the expert distribution masks for each layer of a given model.

  Args:
    model (nn.Module): The model for which to plot the expert distribution masks.
    images (List): A list of images to visualize.
    subset: The subset of images to visualize.
    transform (Optional[None], optional): An optional transformation to apply to the images. Defaults to None.
    save_dir (str, optional): The directory to save the generated plots. Defaults to None.
  """
  
  model.eval()
  device = get_model_device(model)
  num_registers = getattr(model, 'num_registers', 0) 
  num_class_tokens = getattr(model, 'num_class_tokens', 1)
  budget_token = getattr(model, 'add_budget_token', False)
  
  image_size = max(images[0][0].shape[-1], images[0][0].shape[0])  # it could be channel first or channel last
  patch_size = model.patch_size
  patches_per_side = (image_size // patch_size)

  for img_idx in tqdm(subset, desc='Preparing masking plots'):

    img, label = images[img_idx]
    # forward pass
    _img = model_transform(img) if model_transform is not None else img
    out = model(make_batch(_img).to(device))
    
    from dataset import IMAGENETTE_CLASSES
    #print(f'Predicted class: {IMAGENETTE_CLASSES[torch.argmax(out).item()]} Ground truth class: {IMAGENETTE_CLASSES[label]}')

    # retrieve last forward masks
    gates = get_forward_masks(model)  # <residual, gating_probs>
    thresolds = get_learned_thresholds(model) # <redsidual, thre>
    
    # prepare plot, we want a row for each residual layer,
    # and two columns, one for the image and one for token masks
    fig, axs = plt.subplots(len(gates.keys())+1, 1, squeeze=False, figsize=(10, 25))

    # plot the image
    img = prepare_for_matplotlib(visualization_transform(img) if visualization_transform is not None else img)
    axs[0,0].imshow(img)

    # for each layer, plot the image and the token mask
    for layer_idx, (layer_name, forward_mask) in enumerate(gates.items()):
      
      if budget_token:
         forward_mask = forward_mask[:, :-1, :]

      forward_mask = forward_mask[:, num_class_tokens+num_registers-1:].detach().reshape(-1, patches_per_side, patches_per_side)  # discard class token and reshape as image
      # replace non-zero values with 1
      if hard:
        forward_mask = torch.nn.functional.relu(forward_mask - thresolds[layer_name])
        # print(torch.any(forward_mask >= 0.5))
        
      forward_mask = prepare_for_matplotlib(forward_mask)
      im = axs[layer_idx+1,0].imshow(forward_mask, vmin=0, vmax=1)
      axs[layer_idx+1,0].title.set_text(layer_name)
      cbar = axs[layer_idx+1,0].figure.colorbar(im, ax=axs[layer_idx+1,0], orientation='horizontal', shrink=0.2)

      # set title to predicted and ground truth class
      title = f'Predicted class: {IMAGENETTE_CLASSES[torch.argmax(out).item()]} Ground truth class: {IMAGENETTE_CLASSES[label]}'
      axs[0,0].title.set_text(title)

    fig.tight_layout()

    if save_dir is not None:
      os.makedirs(save_dir, exist_ok=True)
      plt.tight_layout()
      plt.savefig(join(save_dir, f'token_masks_batch_{img_idx}.jpg'), dpi=100)
    plt.close()



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


######################################################## Old ##################################################################

"""@torch.no_grad()
def expert_distribution(model, images: List, save_dir: str=None):


    model.eval()
    device = get_model_device(model)

    image_size = max(images[0].shape[-1], images[0].shape[0]) # it could be channel first or channel last
    patch_size = model.patch_size
    patches_per_side = (image_size // patch_size)

    for img, label in images:  

        # forward pass      
        img = make_batch(img)
        model(img.to(device))
        
        # retrieve last forward gating probs
        gates = get_last_forward_gates(model)  # <moe, gating_probs>

        # for each moe layer, plot the distribution of experts
        for moe_name, gating_probs in gates.items():
            
            # we assume top-1 gating  
            # gating probs is batch, tokens, exp
            max_vals, max_idx = gating_probs.max(dim=-1)
            batch_masks = max_idx[:, 1:].reshape(-1, patches_per_side, patches_per_side)  # discard class token and reshape as image

            # prepare plot, each row is an image, first column is the image, second column is the expert distribution
            fig, axs = plt.subplots(img.shape[0], 2, squeeze=False)

            # for each image
            for i in range(img.shape[0]):
                img, mask = img[i], batch_masks[i]
                img, mask = prepare_for_matplotlib(img), prepare_for_matplotlib(mask)
                axs[i, 0].imshow(img)
                axs[i, 1].imshow(mask)
            
            plt.suptitle(moe_name)
            if save_dir is not None:
                subset_save_dir = f'{save_dir}/{moe_name.split(".")[-2]}/'
                os.makedirs(subset_save_dir, exist_ok=True)
                plt.savefig(subset_save_dir+f'expert_distribution_batch_{i}.jpg')"""



"""@torch.no_grad()
def token_distribution(model, loader, save_dir=None, image_size = None):
  # TODO not working for now
  model.eval()
  device = get_model_device(model)

  patch_size = model.patch_size
  if image_size is None:
    image_size = model.image_size

  patch_size = model.patch_size
  patches_per_side = (image_size // patch_size)



  for elem, label in tqdm(loader):
    batch = make_batch(elem)
    model(batch.to(device))

    # get dictionary <moe_name, gating_probs> where gating probs is batch, tokens, exp
    gates = get_last_forward_gates(model) 

    # reshape image as sequence of tokens
    unrolled_batch = rearrange(batch, 
                               "b c (h s1) (w s2) -> b (h w) s1 s2 c ", 
                               s1=patch_size, s2=patch_size) # b, tokens, patch_size, patch_size, c

    for moe_name, gating_probs in gates.items():
      
      # get token-expert assignment by maximizing token prob
      max_vals, max_idx = gating_probs.max(dim=-1) # max_vals is a (b, tokens)
      # discard class token 
      batch_masks = max_idx[:, 1:] 
    

      fig, axs = plt.subplots(batch.shape[0], 2, squeeze=False)


      # for each image
      for i in range(batch.shape[0]):
        width = batch[0].shape[-1]
        print(width)
        tokens, mask = unrolled_batch[i], batch_masks[i] # tokens, patch_size, patch_size, channels - tokens
        print(tokens.shape, mask.shape)
        sorted_mask, sorted_idx = torch.sort(mask)
        print(mask)
        print(sorted_idx)
        sorted = torch.index_select(tokens, 0, sorted_idx)
        #sorted = sorted.reshape(width, width, -1)
        # sorted = rearrange(sorted, 't a b c ->  (t a) b c)') # 16, 4, 4, 3 -> 
        sorted = rearrange(sorted, 'b1 h w c -> h (b1 w) c', b1=patches_per_side**2)
        # sorted = rearrange(sorted, 'b1 h w c -> h (b1 w)  c', b1=patch_size)
        axs[i, 0].imshow(sorted)
        axs[i, 1].imshow(sorted_mask.reshape(-1, patches_per_side*patches_per_side)) #.permute(1,0))
      
            
      plt.suptitle(moe_name)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{moe_name.split(".")[-2]}/token_distribution_batch_{i}.jpg')"""