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


def solo_mse(model, 
             budget: float = 0.65, 
             strict: bool = False, 
             skip_layers: List = [],  
             per_layer: bool = True,
             **kwargs):
    
    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    for layer, (_, mask) in enumerate(masks.items()):
        if layer not in skip_layers: 
            sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask, for each image in the batch
            
            # if per layer, we compute MSE for each layer 
            if per_layer:
                sparsity = torch.sum((sparsity - budget) ** 2 if strict else (relu(sparsity - budget))**2)
            sparsity_loss.append(sparsity)
             
    
    sparsity_loss = torch.stack(sparsity_loss)


    if not per_layer:
        # if not per layer, we average the sparsity across all layers and then compute the MSE 
        sparsity_loss = torch.mean(sparsity_loss)
        sparsity_loss = torch.sum((sparsity_loss - budget) ** 2 if strict else (relu(sparsity_loss - budget))**2)

    return torch.mean(sparsity_loss) * (2-budget)


def avit_ponder_loss(model, **kwargs):
    """
    Computes the ponder loss of the model.

    Args:
        model: The model for which to compute the ponder loss.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The ponder loss.
    """
    ponder_loss = torch.mean(model.rho_token)

    return ponder_loss


def avit_distr_prior_loss(model, target_depth=7, **kwargs):
    """
    Computes the distribution prior loss of the model.

    Args:
        model: The model for which to compute the distribution prior loss.
        log_distr_target: The target distribution to compare the model's distribution to.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The distribution prior loss.
    """
    target_dist = torch.distributions.Normal(loc=target_depth, scale=1.0)
    target_dist = target_dist.log_prob(torch.arange(model.num_layers) + 1)
    halting_score_distr = model.halting_score_distr
    halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
    halting_score_distr = torch.clamp(halting_score_distr, 0.001, 0.999)
    distr_prior_loss = torch.nn.functional.kl_div(halting_score_distr.log(),
                                                    target_dist.to(halting_score_distr.device).detach(),
                                                    reduction='batchmean',
                                                    log_target=True)

    return  distr_prior_loss



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

    def __init__(self, budget: float = None, strict: bool = False, skip_layers : List = [], per_layer:bool = True, **kwargs) -> None:
        super().__init__()
        self.budget = budget
        self.strict = strict
        self.skip_layers = skip_layers
        self.per_layer = per_layer

    def forward(self, model, budget = None, per_layer: bool = None, **kwargs):
        """
        Computes the MSE loss of the model.

        Args:
            model (nn.Module): The model to compute the MSE loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The MSE loss.
        """
        assert budget or self.budget, 'budget must be provided either as argument or as class attribute'
        per_layer = per_layer if per_layer is not None else self.per_layer
        return solo_mse(model, budget if budget is not None else self.budget, self.strict, skip_layers=self.skip_layers, per_layer=per_layer)


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


class AViTPonderLoss(ResidualModelLoss):
    """
    Computes the ponder loss of the model.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model, **kwargs):
        """
        Computes the ponder loss of the model.

        Args:
            model (nn.Module): The model to compute the ponder loss for.
            **kwargs: Additional arguments.

        Returns:
        torch.Tensor: The ponder loss.
        """
        return avit_ponder_loss(model)

class AViTDPriorLoss(ResidualModelLoss):
    """
    Computes the distribution prior loss of the model.
    """

    def __init__(self, target_depth: int) -> None:
        super().__init__()
        self.target_depth = target_depth

    def forward(self, model, **kwargs):
        """
        Computes the distribution prior loss of the model.

        Args:
            model (nn.Module): The model to compute the distribution prior loss for.
            **kwargs: Additional arguments.

        Returns:
        torch.Tensor: The distribution prior loss.
        """
        return avit_distr_prior_loss(model, target_depth=self.target_depth)


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


