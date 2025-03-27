"""
Gradient reversal layer for pytorch module

Reference:
    Adapted from https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
"""

from torch import tensor
from torch.autograd import Function

class ReverseLayerF(Function):
    """
    Custom autograd class that reverses and scales gradients during backward pass.

    During forward:
    - Acts as identity operation (returns input unchanged)

    During backward:
    - Reverses gradient direction by multiplying with negative alpha
    - Scales gradients by alpha coefficient

    Args:
        input_ (Tensor): Input tensor for forward pass
        alpha_ (float): Gradient scaling factor, defaults to 1.0
    """

    @staticmethod
    def forward(ctx, input_, alpha_=1.0):
        alpha_ = tensor(alpha_, requires_grad=False)
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class NeutralizeLayerF(Function):
    """
    Custom autograd Function that nullifies gradients during backward pass.

    During forward:
    - Acts as identity operation (returns input unchanged)

    During backward:
    - Returns zero gradients, stopping gradient flow

    Args:
        input_ (Tensor): Input tensor for forward pass
    """

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = 0.0
        return grad_input, None
