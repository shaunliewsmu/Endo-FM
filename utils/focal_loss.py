import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Focal Loss was proposed in "Focal Loss for Dense Object Detection" 
    (https://arxiv.org/abs/1708.02002) and is particularly useful for 
    imbalanced datasets as it down-weights well-classified examples.
    
    Args:
        alpha (float or list): Weighting factor for each class
            - If float: Weight for positive class (1-alpha for negative class)
            - If list: Class-specific weights
        gamma (float): Focusing parameter (higher means more focus on hard examples)
        reduction (str): 'none', 'mean', or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape (N, C) where C is the number of classes
            targets: Tensor of shape (N) where each value is the class index
            
        Returns:
            Calculated loss
        """
        # Get probabilities
        log_softmax = F.log_softmax(inputs, dim=1)
        logpt = log_softmax.gather(1, targets.view(-1, 1))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
            
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                # Convert alpha list to tensor and get alpha for each target
                alpha_t = torch.tensor([self.alpha[t.item()] for t in targets], 
                                     device=inputs.device)
                focal_weight = alpha_t * focal_weight
            else:
                # For binary classification - alpha for positive class, 1-alpha for negative
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_weight = alpha_t * focal_weight
                
        # Apply focal weight to cross entropy loss
        loss = -focal_weight * logpt
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss