import torch.optim as optim


def make_optimizer(model, lr: float, betas: tuple[float, float], weight_decay: float):
    """ Fused AdamW over all of [model]'s parameters """
    return optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        fused=True,
        betas=betas,
    )
