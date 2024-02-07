from collections import defaultdict
from omegaconf import OmegaConf,DictConfig
import torch
from .utils import get_model_device, get_forward_masks
from einops import reduce
from torch.nn.functional import cross_entropy, relu
from torch.special import entr
from abc import ABC, abstractmethod
from typing import List, Literal
from hydra.utils import instantiate



class ResidualModelLoss(torch.nn.Module, ABC):
    
    @abstractmethod
    def forward(self, model, **kwargs):
        pass


####################################################### functional implementations ##################################################################


def sparsity_loss_per_block(model, budget: float = 0.65, sparsity_type : Literal['l1', 'mse', 'cross_entropy' ] = 'l1', **kwargs):
    """
    Computes the sparsity loss per block.

    Args:
        model: The model for which to compute the sparsity loss.
        budget (float): The desired sparsity level.
        sparsity_type (str): The type of sparsity loss to compute, 
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple: A tuple containing the mean sparsity loss and the mean intra-entropy.
    """
    
    masks = get_forward_masks(model)

    # compute the sparsity loss
    sparsity_loss = []
    intra_entropy = []
    for _, mask in masks.items():
        # compute the sparsity loss for each sequence in the batch 
        # mask only contains 1s and 0s, so the mean is the sparsity
        sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask
        
        if sparsity_type == 'mse':
            # force sparsity to be close to budget with mse
            sparsity_loss.append(torch.mean((sparsity - budget) ** 2))
        if sparsity_type == 'cross_entropy':
            # force sparsity to be close to budget with cross entropy
            sparsity_loss.append(cross_entropy(sparsity, torch.tensor([budget] * sparsity.shape[0]).to(get_model_device(model))))
        elif sparsity_type == 'l1':
            # force sparsity to be close to budget with l1
            sparsity_loss.append(torch.mean(torch.abs(sparsity - budget)))

        # force sparsity inside image by maximizing entropy
        # print(sparsity)
        intra_entropy.append(entr(sparsity))
        
    sparsity_loss = torch.stack(sparsity_loss)


    return torch.mean(sparsity_loss)


def entropy_per_blocks(model, **kwargs):

    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # compute the sparsity loss
    intra_entopy = []
    for _, mask in masks.items():
        # compute the sparsity loss for each mask
        sparsity = reduce(mask, 'b s 1 -> b', 'mean')
        intra_entopy.append(entr(sparsity))
    
    intra_entopy = torch.stack(intra_entopy)

    return torch.mean(intra_entopy), 


def solo_l1(model, budget: float = 0.25, strict:bool = False, **kwargs):

    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    for _, mask in masks.items():
        sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask
        sparsity_loss.append(torch.sum(torch.abs(sparsity - budget)))
    
    sparsity_loss = torch.stack(sparsity_loss)

    return torch.mean(sparsity_loss)


def solo_mse(model, budget: float = 0.65, strict: bool = False, skip_layers: List = [], **kwargs):
    
    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    for layer, (_, mask) in enumerate(masks.items()):
        if layer not in skip_layers: 
            sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask
            # if budget > 0.9 : strict = True # TODO quick test
            sparsity = torch.sum((sparsity - budget) ** 2 if strict else (relu(sparsity - budget))**2)
            sparsity_loss.append(sparsity)
            # sparsity_loss.append(sparsity + (1-budget) / 1e1) # TODO added this correction term here 
    
    sparsity_loss = torch.stack(sparsity_loss)

    return torch.mean(sparsity_loss)


def l1_and_intraentropy(model, budget: float = 0.65,  **kwargs):

    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    intra_entropy = []
    for _, mask in masks.items():
        sparsity = reduce(mask, 'b s 1 -> b', 'mean')
        sparsity_loss.append(torch.sum(torch.abs(relu(sparsity - budget))))
    
        intra_entropy.append(entr(sparsity))
    
    sparsity_loss = torch.stack(sparsity_loss)

    return torch.mean(sparsity_loss)


####################################################### class implementations ##################################################################

class SparsityLoss(ResidualModelLoss):
    """
    Computes the sparsity loss of the model.
    """

    def __init__(self, budget: float) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model, budget=None, **kwargs):
        """
        Computes the sparsity loss of the model.

        Args:
            model (nn.Module): The model to compute the sparsity loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The sparsity loss.
        """
        return sparsity_loss_per_block(model, budget= budget or self.budget, **kwargs)


class EntropyLoss(ResidualModelLoss):
    """
    Computes the entropy loss of the model.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model, **kwargs):
        """
        Computes the entropy loss of the model.

        Args:
            model (nn.Module): The model to compute the entropy loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The entropy loss.
        """
        return entropy_per_blocks(model)


class L1Loss(ResidualModelLoss):
    """
    Computes the L1 loss of the model.
    """

    def __init__(self, budget: float) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model, budget = None, **kwargs):
        """
        Computes the L1 loss of the model.

        Args:
            model (nn.Module): The model to compute the L1 loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The L1 loss.
        """
        # if batch is not None and a budget was not provided, compute the budget from the batch
        return solo_l1(model, budget or self.budget)


class MSELoss(ResidualModelLoss):
    """
    Computes the MSE loss of the model.
    """

    def __init__(self, budget: float = None, strict: bool = False, skip_layers : List = [], **kwargs) -> None:
        super().__init__()
        self.budget = budget
        self.strict = strict
        self.skip_layers = skip_layers

    def forward(self, model, budget = None, **kwargs):
        """
        Computes the MSE loss of the model.

        Args:
            model (nn.Module): The model to compute the MSE loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The MSE loss.
        """
        assert budget or self.budget, 'budget must be provided either as argument or as class attribute'
        
        return solo_mse(model, budget if budget is not None else self.budget, self.strict, skip_layers=self.skip_layers)


class ChannelMSELoss(ResidualModelLoss):
    """
    Computes the MSE loss of the model. It is a copy of MSELoss with a different name. 
    The reason for the different name is that this loss is supposed to be used for channel bandwith and not for general model budget.
    """

    def __init__(self, budget: float = None, strict: bool = False, skip_layers : List = [], **kwargs) -> None:
        super().__init__()
        self.budget = budget
        self.strict = strict
        self.skip_layers = skip_layers

    def forward(self, model, channel_budget = None, **kwargs):
        """
        Computes the MSE loss of the model.

        Args:
            model (nn.Module): The model to compute the MSE loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The MSE loss.
        """
        assert channel_budget or self.budget, 'budget must be provided either as argument or as class attribute'
        
        return solo_mse(model, channel_budget if channel_budget is not None else self.budget, self.strict, skip_layers=self.skip_layers)

class L1AndIntraEntropyLoss(ResidualModelLoss):
    """
    Computes the L1 loss and the intra-entropy of the model.
    """

    def __init__(self, budget: float) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model, budget = None, **kwargs):
        """
        Computes the L1 loss and the intra-entropy of the model.

        Args:
            model (nn.Module): The model to compute the L1 loss and the intra-entropy for.
            **kwargs: Additional arguments.

        Returns:
            Tuple: A tuple containing the L1 loss and the intra-entropy.
        """
        return l1_and_intraentropy(model, budget or self.budget)


class AlwaysZeroLoss(ResidualModelLoss):
    """
    A loss function that always returns zero.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model, **kwargs):
        return torch.tensor(0.0), torch.tensor(0.0)


class LossCompose:
    """
    A class that composes multiple loss functions together.

    Args:
        losses_dict (dict): A dictionary containing the loss functions and their arguments.
        Notice that each element in the dictionary must be a dictionary containing 
        at least the key _target_ that points to a class that can be instantiated by hydra. 

    Attributes:
        additional_losses (defaultdict): A dictionary that stores the additional losses with their weights and loss functions. See losses yaml file for example.

    Methods:
        compute: Computes the total loss by evaluating each additional loss function.

    """

    def __init__(self, losses_dict):
        """
        Initializes the LossCompose object.

        Args:
            losses_dict (dict): A dictionary containing the loss functions and their arguments.

        """
        if isinstance(losses_dict, DictConfig):
            losses_dict = OmegaConf.to_container(losses_dict, resolve=True)

        self.additional_losses = defaultdict(dict)  
        for loss, loss_args in losses_dict.items():
            self.additional_losses[loss]['weight'] = loss_args.pop('weight', 1.)
            self.additional_losses[loss]['loss_fn'] = instantiate(loss_args)
        

    def compute(self, model, dict_prefix='', return_dict=True, **kwargs):
        """
        Computes the total loss by evaluating each additional loss function.

        Args:
            model: The model used for computing the loss.
            dict_prefix (str): A prefix to be added to the loss names in the losses_dict.
            return_dict (bool): Whether to return the losses_dict along with the total loss.
            **kwargs: Additional keyword arguments to be passed to the loss functions.

        Returns:
            total_loss: The computed total loss.
            (optional) losses_dict: A dictionary containing the individual losses and their values. 

        """
        losses_dict = {}
        total_loss = []
        for loss, loss_args in self.additional_losses.items():
            l = loss_args['loss_fn'](model, **kwargs) * loss_args['weight']
            losses_dict[f'{dict_prefix}{loss}'] = l.detach().item()
            total_loss.append(l)
        total_loss = torch.stack(total_loss).sum()
        if return_dict:
            return losses_dict, total_loss
        else:
            return total_loss


