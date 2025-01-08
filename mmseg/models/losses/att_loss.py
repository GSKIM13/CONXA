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


def attention_loss(output, target, opps_target):
    num_pos = torch.sum(target == 1).float() + torch.sum(opps_target == 1).float()
    num_neg = torch.sum(target == 0).float() + torch.sum(opps_target == 0).float()
#    num_pos = torch.sum(target == 1).float()
#    num_neg = torch.sum(target == 0).float()

    alpha = num_neg / (num_pos + num_neg) * 1.0

#    print(num_neg)
#    print(num_pos + num_neg)
#    print(alpha)

    eps = 1e-14
#    weight = target * alpha + (1.0 - target) * (1.0 - alpha)
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5))/4 + \
            (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))/4
    weight = weight.detach()

    loss = F.binary_cross_entropy(output, target.float(), weight, reduction='none')
    loss = torch.sum(loss)
    return loss


def att_loss(pred,
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
        p = pred[b_i, :, :, :]
        t = label[b_i, :, :, :].squeeze()
#        print(p.size())
#        print(t.size())
#        exit()
        t = (t > 0.5).float()
#        b, c, h, w = mask.shape

        all_edges = torch.clamp((t[0] + t[1] + t[2] + t[3]), min=0.0, max=1.0)

        opps_targ = all_edges - t[0]
        loss_cls0 = attention_loss(p[0], t[0], opps_targ)

        opps_targ = all_edges - t[1]
        loss_cls1 = attention_loss(p[1], t[1], opps_targ)

        opps_targ = all_edges - t[2]
        loss_cls2 = attention_loss(p[2], t[2], opps_targ)

        opps_targ = all_edges - t[3]
        loss_cls3 = attention_loss(p[3], t[3], opps_targ)

        total_loss = total_loss + loss_cls0 + loss_cls1 + loss_cls2 + loss_cls3

    total_loss = total_loss / batch
    return total_loss



@LOSSES.register_module()
class ATTLoss(nn.Module):
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
        super(ATTLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.cls_criterion = att_loss

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
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
