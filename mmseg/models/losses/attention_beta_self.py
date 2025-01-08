import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

@LOSSES.register_module()
class Att_Beta_Self_LOSS(nn.Module):
    """HEDLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(Att_Beta_Self_LOSS, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.channel_weights = nn.Parameter(torch.ones(4, dtype=torch.float))
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def att_self_loss(self, pred,
                      label,
                      weight=None,
                      reduction='mean',
                      avg_factor=None,
                      class_weight=None):
        """Calculate the binary CrossEntropy loss with weights.
        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.

        Returns:
            torch.Tensor: The calculated loss
        """
        if weight is not None:
            weight = weight.float()

        total_loss = 0

        label = label.unsqueeze(1)
        batch, channel_num, imh, imw = pred.shape

        for b_i in range(batch):
            for c in range(4):
                p = pred[b_i, c, :, :].unsqueeze(0)
                t = label[b_i, :, c, :, :]

                # Ensure p and t are valid
                if torch.isnan(p).any() or torch.isinf(p).any():
                    raise ValueError(f"Invalid value in prediction: {p}")
                if torch.isnan(t).any() or torch.isinf(t).any():
                    raise ValueError(f"Invalid value in target: {t}")

                mask = (t > 0.5).float()

                b, h, w = mask.shape
                num_pos = torch.sum(mask, dim=[1, 2]).float()
                num_neg = h * w - num_pos

                alpha = num_neg / (num_pos + num_neg + 1e-6) * 1.0
                class_weight = torch.zeros_like(mask)
                class_weight[t > 0.5] = num_neg / (num_pos + num_neg + 1e-6)
                class_weight[t <= 0.5] = num_pos / (num_pos + num_neg + 1e-6)

                # Ensure alpha and class_weight are valid
                if torch.isnan(alpha).any() or torch.isinf(alpha).any():
                    raise ValueError(f"Invalid value in alpha: {alpha}")
                if torch.isnan(class_weight).any() or torch.isinf(class_weight).any():
                    raise ValueError(f"Invalid value in class_weight: {class_weight}")

                eps = 1e-6
                p_clip = torch.clamp(p, min=eps, max=1.0 - eps)

                # Ensure all tensors are on the same device
                device = p.device
                alpha = alpha.to(device)
                self.channel_weights = self.channel_weights.to(device)
                weight = t * alpha * (self.channel_weights[c] ** ((1.0 - p_clip) ** 0.5)) + \
                         (1.0 - t) * (1.0 - alpha) * (self.channel_weights[c] ** (p_clip ** 0.5))

                # Clamping weight to avoid extreme values
                weight = torch.clamp(weight, min=eps, max=1e6)
                
                # Ensure weight is valid
                if torch.isnan(weight).any() or torch.isinf(weight).any():
                    raise ValueError(f"Invalid value in weight: {weight}")


                loss = self.bce_loss(p, t.float()) * weight
                
                
                
                total_loss += loss.sum()
                
                total_loss += 1e3 / self.channel_weights[c]

        if self.reduction == 'mean' and avg_factor is not None:
            total_loss = total_loss / avg_factor
            print('mean')
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
            print('sum')

        #print(self.channel_weights)

        #total_loss += 1e7 / (torch.prod(self.channel_weights) + eps)

        return total_loss * self.loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.att_self_loss(
            cls_score,
            label,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor)

        total_loss = loss_cls

        total_loss = total_loss * self.loss_weight

        return total_loss
