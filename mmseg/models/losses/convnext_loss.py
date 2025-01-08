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
class ConvNeXtLoss(nn.Module):
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
                 attention_coeff=1.0,
                 gamma = 0.5,
                 beta = 8,
                 dice_coeff = 2500,
                 rev_dice_coeff = 2500,
                 iou_coeff = 0):
        super(ConvNeXtLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.attention_coeff = attention_coeff
        self.class_weight = class_weight
        self.gamma = gamma
        self.beta = beta
        self.dice_coeff = dice_coeff
        self.rev_dice_coeff = rev_dice_coeff
        self.iou_coeff = iou_coeff

        self.cls_criterion = self.hed_attention_loss
        
    def hed_attention_loss(self, pred,
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
          p_all = pred[b_i,:,:,:].unsqueeze(0) #1,4,320,320
          t_all = label[b_i,:,:,:,:] #1,4,320,320
          
          b, c, h, w = t_all.shape
          
          num_pos = torch.sum(t_all, dim=[1,2,3]).float()
          num_neg = c*h*w - num_pos
          
          alpha = num_neg / (num_pos+num_neg)*1.0
          
          for c in range(4):
            p = pred[b_i, c, :, :].unsqueeze(0)
            t = label[b_i, :, c, :, :]
            
            eps = 1e-14
            p_clip = torch.clamp(p, min=eps, max=1.0 - eps)

            weight = t * alpha * (self.beta ** ((1.0 - p_clip) ** self.gamma)) + \
             (1.0 - t) * (1.0 - alpha) * (self.beta ** (p_clip ** self.gamma))
            weight=weight.detach()
               
            loss = F.binary_cross_entropy(p, t.float(), weight=weight, reduction='none')

            loss = torch.sum(loss)
            total_loss = total_loss + loss

        return total_loss
        
    def dice_loss_old(self, pred, target, smooth=1e-6):
        pred = pred.contiguous()
        target = target.contiguous()    
    
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
        return loss.sum()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous()
        target = target.contiguous()    
    
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + smooth) / ((pred**2).sum(dim=2).sum(dim=2) + (target**2).sum(dim=2).sum(dim=2) + smooth)))
        
        return loss.sum()
        
    def iou_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        union = (pred + target).sum(dim=2).sum(dim=2) - intersection
        
        loss = 1 - ((intersection + smooth) / (union + smooth))
        
        return loss.sum()



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
        loss_cls = self.attention_coeff * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)
            
        dice = self.dice_coeff *self.dice_loss(cls_score, label)
        rev_dice = self.rev_dice_coeff * self.dice_loss(1-cls_score, 1-label)
        
        iou = self.iou_coeff * self.iou_loss(cls_score, label)
        
                
        loss_cls += dice
        loss_cls += rev_dice 
        loss_cls += iou
        
        return loss_cls
