import torch
import torch.nn as nn



def get_param_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    whitelist_weight_modules: tuple = (nn.Linear,),
    blacklist_weight_modules: tuple = (nn.LayerNorm, nn.Embedding),
) -> list:
    """
    Returns parameter groups for weight decay / no weight decay.

    Args:
        model (nn.Module): The PyTorch model whose parameters we want to split.
        weight_decay (float): Weight decay for eligible parameters.
        whitelist_weight_modules (tuple): Modules whose weights get weight decay
                                          (e.g., nn.Linear).
        blacklist_weight_modules (tuple): Modules whose weights do NOT get weight
                                          decay (e.g., nn.LayerNorm, nn.Embedding, RMSNorm).

    Returns:
        list: A list of parameter group dicts to pass to an optimizer, e.g. torch.optim.AdamW.
    """
    decay_params = set()
    no_decay_params = set()

    # Collect references to all parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    trainable_params = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # trainable_params = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # Go through each sub-module to decide which params get decay or not
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            # Skip parameters that are not trainable


            # Biases -> no decay
            if param_name.endswith("bias") or "norm" in full_param_name or "cls_token" in full_param_name:
                no_decay_params.add(full_param_name)
            # Whitelisted modules -> decay
            elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                decay_params.add(full_param_name)
            # Blacklisted modules -> no decay
            elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                no_decay_params.add(full_param_name)
            elif param_name.endswith("in_proj_weight"):
                decay_params.add(full_param_name)

    # Ensure consistency
    inter_params = decay_params & no_decay_params
    if len(inter_params) > 0:
        raise ValueError(f"Parameters in both decay and no_decay sets: {inter_params}")

    union_params = decay_params | no_decay_params
    missing_params = set(trainable_params.keys()) - union_params
    if len(missing_params) > 0:
        raise ValueError(f"Some params not in decay or no_decay sets: {missing_params}")

    # Build parameter groups
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(decay_params)],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(no_decay_params)],
            "weight_decay": 0.0,
        },
    ]
    return optim_groups

