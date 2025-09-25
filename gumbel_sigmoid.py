import torch
from torch import Tensor
from torch.distributions.gumbel import Gumbel


def gumbel_sigmoid(logits: Tensor, temp: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:

    """
    Samples from the Gumbel-Sigmoid distribution with an optional hard gate at the selected threshold.
    
    Gumbel Sigmoid is equivalent to gumbel softmax for two classes with one class being 0
    i.e. gumbel_sigmoid = e^([a+gumbel1]/t) / [e^([a+gumbel1]/t) + e^(gumbel2/t)] = sigm([a+gumbel1-gumbel2]/t)

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      temp: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    # Temperature must be positive.
    if temp <= 0:
        raise ValueError("Temperature must be positive")

    # Sample Gumbel noise. The difference of two Gumbels is equivalent to a Logistic distribution.
    gumbel_noise = Gumbel(0, 1).sample(logits.shape).to(logits.device) - \
                   Gumbel(0, 1).sample(logits.shape).to(logits.device)
    
    # Apply the reparameterization trick
    y_soft = torch.sigmoid((logits + gumbel_noise) / temp)

    if hard:
        # Straight-Through Estimator
        y_hard = (y_soft > threshold).float()
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft
