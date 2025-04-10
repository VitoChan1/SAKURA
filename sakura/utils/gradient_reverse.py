"""
Gradient reversal layer for pytorch module

Reference:
    Adapted from https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
"""

from torch import tensor
from torch.autograd import Function

class ReverseLayerF(Function):
    """
    Custom autograd class that reverses and scales gradients during backward pass

    During forward:
        - Acts as identity operation (returns input unchanged)
    During backward:
        - Reverses gradient direction by multiplying with negative alpha;
        - Scales gradients by alpha coefficient

    """

    @staticmethod
    def forward(ctx, input_, alpha_= 1.0):
        """
        Forward pass for a custom autograd operation with gradient scaling.

        :param ctx: Context object to save tensors for backward computation
        :type ctx: torch.autograd.function.FunctionCtx
        :param `input_`: Input tensor of shape (N, \*) for forward pass, where \* means number of dimensions
        :type `input_`: torch.Tensor
        :param `alpha_`: Gradient scaling factor, defaults to 1.0 (no scaling)
        :type `alpha_`: float, optional

        :return: Output tensor identical to `input_` (shape preserved)
        :rtype: torch.Tensor
        """
        alpha_ = tensor(alpha_, requires_grad=False)
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        """
        Backward pass for the custom autograd operation with gradient scaling.

        :param ctx: Context object containing saved tensors from forward pass
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_output: Upstream gradient of shape (N, \*), matching the forward input dimensions
        :type grad_output: torch.Tensor
        :return:
            - grad_input: Gradient of `input_` scaled by -`alpha_` (shape preserved)
            - None: Placeholder for alpha gradient (not calculated)

        :rtype: tuple (torch.Tensor, None)
        """
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class NeutralizeLayerF(Function):
    """
    Custom autograd Function that nullifies gradients during backward pass

    During forward:
        - Acts as identity operation (returns input unchanged)
    During backward:
        - Returns zero gradients, stopping gradient flow

    """

    @staticmethod
    def forward(ctx, input_):
        """
        Forward pass for gradient neutralization layer (identity transform).

        :param ctx: Context object containing saved tensors from forward pass
        :type ctx: torch.autograd.function.FunctionCtx
        :param `input_`: Input tensor of shape (N, \*) for forward pass, where \* means number of dimensions
        :type `input_`: torch.Tensor

        :return: Output tensor identical to `input_` (shape preserved)
        :rtype: torch.Tensor
        """
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        """
        Backward pass that nullifies upstream gradients.

        :param ctx: Context object containing saved tensors from forward pass
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_output: Upstream gradient of shape (N, \*), matching the forward input dimensions
        :type grad_output: torch.Tensor
        :return:
            - grad_input: Zero-valued gradient tensor
            - None: Placeholder for unused gradient
        :rtype: tuple (torch.Tensor, None)
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = 0.0
        return grad_input, None
