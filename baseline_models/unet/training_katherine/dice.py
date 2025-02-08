import torch

def dice_coefficient(pred, truth, threshold=0.5, flip=False):
    """
    Compute the Dice coefficient.
    
    Parameters:
    - pred (torch.Tensor): Predictions (probabilities) from the model.
    - truth (torch.Tensor): Ground truth binary mask.
    - threshold (float): Threshold to convert probabilities into binary predictions.
    
    Returns:
    - dice (float): Computed Dice coefficient.
    """
    # Convert probabilities to binary predictions
    pred_bin = (pred >= threshold).float()
    
    # Calculate intersection and union
    if flip:
        pred_bin = 1 - pred_bin
        truth = 1 - truth
        intersection = (pred_bin * truth).sum()
        union = pred_bin.sum() + truth.sum()
    else:
        intersection = (pred_bin * truth).sum()
        union = pred_bin.sum() + truth.sum()
    
    # Calculate Dice coefficient
    dice = (2. * intersection + 1e-6) / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
    
    return 1 - dice