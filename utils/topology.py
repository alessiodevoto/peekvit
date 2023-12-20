import torch
from einops import reduce




def add_class_tokens(x, num_class_tokens):
    pass


def add_register_tokens(model, num_registers):
    model.register_tokens = torch.nn.Parameter(torch.zeros(1, num_registers, model.hidden_dim))
    model.seq_length += num_registers

    num_class_tokens = getattr(model, 'num_class_tokens', 1)
    model.num_special_tokens = num_class_tokens + num_registers

    # we now have to update the model forward to add the register tokens
    def forward(model, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = model._process_input(x)
        n = x.shape[0]

        # Add registers
        if model.num_registers > 0:
            batch_register_tokens = model.register_tokens.expand(n, -1, -1)
            x = torch.cat([x, batch_register_tokens], dim=1)
        
        # Expand the class token to the full batch
        batch_class_tokens = model.class_tokens.expand(n, -1, -1)
        x = torch.cat([x, batch_class_tokens], dim=1)

        # Pass through the encoder
        x = model.encoder(x)

        # Get all class tokens and average them
        x = x[:, 0:model.num_class_tokens]
        x = reduce(x, 'n c e -> n e', reduction='sum')

        # Classification head
        x = model.head(x)

        return x
    
    # replace the forward function
    model.forward = forward
    


def add_residual_gates(residualvit_model, residual_gates_args):
    """
    Adds residual gates to the ResidualViT model.
    The model must be a ResidualViT.

    Args:
        residualvit_model (ResidualViT): The ResidualViT model to add residual gates to.
        residual_gates_args (dict): A dictionary containing the arguments for adding residual gates.
            - 'residual_layers' (list): List of layer names to add residual gates to.
            - 'gate_type' (str): Type of residual gate to add.
            - 'add_input' (bool): Whether to add the input to the residual gate.
            - 'gate_temp' (float): Temperature parameter for the residual gate.

    Returns:
        ResidualViT: The ResidualViT model with residual gates added.
    """

    from models.residualvit import ResidualGate, ResidualViTBlock
    skip = residual_gates_args['residual_layers']
    gate_type = residual_gates_args['gate_type']
    add_input = residual_gates_args['add_input']
    temp = residual_gates_args['gate_temp']
    i = 0
    for module_name, module in residualvit_model.named_modules():
        if isinstance(module, ResidualViTBlock) and skip[i] in {'attention+mlp', 'attention', 'mlp'}:
            print(f'Adding residual gate to {module_name}')
            module.skip = skip[i]  
            module.add_input = add_input
            module.residual_gate = ResidualGate(module.hidden_dim, temp=temp, gate_type=gate_type)
            i += 1
    return residualvit_model



def reinit_class_tokens(model):
    """
    Reinitializes the class tokens in the given model. Assumes the class token parameter has a name containing 'class'.

    Args:
        model (torch.nn.Module): The model to reinitialize the class tokens in.

    Returns:
        torch.nn.Module: The model with reinitialized class tokens.
    """
    for param_name, param in model.named_parameters():
        if 'class' in param_name:
            print(f'Reinitializing {param_name}...', end=' ')
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            print('Reinitialized!')
    return model