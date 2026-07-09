# FILE: ./DGLER/core/losses.py

import torch
import torch.nn.functional as F

def compute_depth_gradient_weight(depth_map):
    """
    Calculate the spatial gradient of the depth map to enhance the weight of optical flow loss in the edge areas.
    depth_map: [B, 1, H, W]
    """
    #  Sobel 
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.],[-1., 0., 1.]], device=depth_map.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=depth_map.device).view(1, 1, 3, 3)
    
    # Filling
    pad_depth = F.pad(depth_map, (1, 1, 1, 1), mode='replicate')
    
    grad_x = F.conv2d(pad_depth, sobel_x)
    grad_y = F.conv2d(pad_depth, sobel_y)
    
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    # Normalize and generate weights (the base weight is 1.0, and the weight increases at the edges)
    # 5.0 represents the scale factor for edge magnification. You can make adjustments based on the experiment.
    grad_mag_norm = grad_mag / (torch.max(grad_mag) + 1e-6)
    edge_weight = 1.0 + 5.0 * grad_mag_norm 
    
    return edge_weight

def sequence_loss(flow_preds, flow_gt, depth_map=None, gamma=0.8, MAX_FLOW=400):
    """ 
    Depth-Gradient Aware Sequence Loss 
    
    """
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2], flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # === Incorporation of edge weights  ===
    edge_weight = 1.0
    if depth_map is not None:
        edge_weight = compute_depth_gradient_weight(depth_map)
        edge_weight = edge_weight.expand(-1, 2, -1, -1) # The number of channels compatible with the flow
        edge_weight = edge_weight * Mask # Only calculate where there is ground truth.

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        
        # Apply geometric depth edge punishment
        if depth_map is not None:
            Loss_reg = Loss_reg * edge_weight
            
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (mask_sum + 1e-5)
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics

def sequence_loss_single(flow_pred, flow_gt, Mask, mask_sum, depth_map=None):
    """ Single-step depth perception loss """
    Loss_reg = (flow_pred - flow_gt) * Mask
    
    if depth_map is not None:
        edge_weight = compute_depth_gradient_weight(depth_map)
        edge_weight = edge_weight.expand(-1, 2, -1, -1)
        Loss_reg = Loss_reg * (edge_weight * Mask)
        
    Loss_reg = torch.norm(Loss_reg, dim=1)
    Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
    Loss_reg = Loss_reg / (mask_sum + 1e-5)
    flow_loss = Loss_reg.mean()

    return flow_loss