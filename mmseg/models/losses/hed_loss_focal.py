import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
class HEDLoss_FOCAL(nn.Module):
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
        super(HEDLoss_FOCAL, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        #self.cls_criterion = self.hed_loss
        self.channel_weights = nn.Parameter(torch.ones(4, dtype=torch.float))
        
      #  self.hed_weight = nn.Parameter(torch.tensor(1.))
      #  self.dice_weight = nn.Parameter(torch.tensor(1.))
      
    def hed_loss_focal(self, pred,
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
    
            #exit()
            mask = (t > 0.5).float()
    
            b, h, w = mask.shape
            #_, b, c, h, w = mask.shape
            num_pos = torch.sum(mask, dim=[1, 2]).float()  # Shape: [b,].
            num_neg = h * w - num_pos  # Shape: [b,].
            
            alpha = num_neg / (num_pos + num_neg) * 1.0
            #print(num_neg)
            #print(num_pos)
            class_weight = torch.zeros_like(mask)
            class_weight[t > 0.5] = num_neg / (num_pos + num_neg)
            class_weight[t <= 0.5] = num_pos / (num_pos + num_neg)
            # weighted element-wise losses
            
            eps = 1e-14
            p_clip = torch.clamp(p, min=eps, max=1.0 - eps)

            weight = t * alpha * (1.0 - p_clip) ** 2 + \
             (1.0 - t) * (1.0 - alpha) * (p_clip ** 2)
            weight=weight.detach()
            
            # print(t.size()) 1,4,320,320
            # print(p.size()) 1,4,320,320
            
            inverse_square_weight = 1.0 / (self.channel_weights[c] ** 2)
            
    
            loss = F.binary_cross_entropy(p, t.float(), weight=weight, reduction='none')
            loss = loss * inverse_square_weight
            #exit()
            # do the reduction for the weighted loss
            #loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
            loss = torch.sum(loss)
            total_loss = total_loss + loss
            #exit()
        total_loss = total_loss + math.log(torch.prod(self.channel_weights))

        #print("finished?")
        return total_loss
        
    def dice_loss(self, pred, target, smooth=1):
        pred = pred.contiguous()
        target = target.contiguous()    
    
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
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
            

        
        loss_cls = self.loss_weight * self.hed_loss_focal(
            cls_score,
            label,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
            
        #dice = self.dice_loss(cls_score.sigmoid(), label)
        
        total_loss = loss_cls# + 1000 * dice
        
        total_loss = total_loss*self.loss_weight
            
            
        return total_loss
