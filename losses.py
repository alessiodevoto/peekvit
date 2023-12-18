import torch
from utils import get_model_device, get_forward_masks
from einops import reduce
from torch.nn.functional import cross_entropy
from torch.special import entr




def sparsity_loss(model, **kwargs):
    
    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # compute the sparsity loss
    sparsity_loss = []
    for _, mask in masks.items():
        # compute the sparsity loss for each mask
        sparsity_loss.append(torch.sum(mask) / mask.numel())
    
    sparsity_loss = torch.stack(sparsity_loss)
    
    return torch.mean(sparsity_loss)


def sparsity_loss_per_block(model, budget: float = 0.65, **kwargs):
    # raise NotImplementedError('Implement this function per block')
    #TODO force each block to have sparsity, if you sum sparisity of each block, it will end up minimizing just one block and 
    # using all the tokens for the other ones.
    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # compute the sparsity loss
    sparsity_loss = []
    entropy_loss = []
    for _, mask in masks.items():
        # compute the sparsity loss for each sequence in the batch 
        # mask only contains 1s and 0s, so the mean is the sparsity
        sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask
        # print(sparsity.mean())
        
        # force sparsity to be close to budget with mse
        # sparsity_loss.append(torch.mean((sparsity - budget) ** 2))
        
        # force sparsity to be close to budget with cross entropy
        # sparsity_loss.append(cross_entropy(sparsity, torch.tensor([budget] * sparsity.shape[0]).to(get_model_device(model))))

        # force sparsity to be close to budget with l1
        sparsity_loss.append(torch.mean(torch.abs(sparsity - budget)))

        # force sparsity inside image by maximizing entropy
        # print(sparsity)
        entropy_loss.append(entr(sparsity))
        


    sparsity_loss = torch.stack(sparsity_loss)
    # print(sparsity_loss)

    entropy_loss = torch.stack(entropy_loss)
    # print(entropy_loss)

    # add entropy here to force the sparsity to be more uniform
    # entropy = - sum_i p_i log(p_i)
    # entr = - torch.sum(sparsity_loss * torch.log(sparsity_loss))
    # print(entr)

    return torch.mean(sparsity_loss), torch.mean(entropy_loss)





LOSSES_MAP = {
    'sparsity': sparsity_loss,
    'sparsity_per_block': sparsity_loss_per_block,
}


def get_loss(loss_type, loss_args):
    """
    Retrieves the loss function with the given type and arguments.

    Args:
        loss_type (str): The type of the loss function.
        loss_args (dict): The arguments of the loss function.

    Returns:
        callable: The loss function.
    """
    if loss_type is None:
        return None
    if loss_type not in LOSSES_MAP:
        raise ValueError(f'Loss type must be one of {LOSSES_MAP.keys()}')
    return LOSSES_MAP[loss_type]