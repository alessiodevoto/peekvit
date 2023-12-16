import torch
from utils import get_model_device, get_forward_masks




def sparsity_loss(model, **kwargs):
    
    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, num_experts)
    masks = get_forward_masks(model)

    # compute the sparsity loss
    sparsity_loss = []
    for _, mask in masks.items():
        # compute the sparsity loss for each mask
        sparsity_loss.append(torch.sum(mask) / mask.numel())
    
    sparsity_loss = torch.stack(sparsity_loss)
    
    return torch.mean(sparsity_loss)

def sparsity_loss(model, **kwargs):
    raise NotImplementedError('Implement this function per block')
    #TODO force each block to have sparsity, if you sum sparisity of each block, it will end up minimizing just one block and 
    # using all the tokens for the other ones.
    return torch.mean(sparsity_loss)




LOSSES_MAP = {
    'sparsity': sparsity_loss
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