# FILE: ./DGLER/core/losses.py

import torch
import torch.nn.functional as F

def compute_depth_gradients(depth_map):
    sobel_x = depth_map.new_tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3).contiguous()
    padded = F.pad(depth_map, (1, 1, 1, 1), mode='replicate')
    return F.conv2d(padded, sobel_x), F.conv2d(padded, sobel_y)


def compute_depth_gradient_weight(depth_map, alpha=5.0, eps=1e-6):
    grad_x, grad_y = compute_depth_gradients(depth_map)
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + eps)
    grad_max = grad_mag.amax(dim=(2, 3), keepdim=True)
    return 1.0 + alpha * grad_mag / (grad_max + eps)


def compute_boundary_normals(depth_ref, flow_gt, eps_g=1e-5):
    height, width = flow_gt.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=flow_gt.device, dtype=flow_gt.dtype),
        torch.arange(width, device=flow_gt.device, dtype=flow_gt.dtype),
        indexing='ij'
    )

    qx = x.unsqueeze(0) + flow_gt[:, 0]
    qy = y.unsqueeze(0) + flow_gt[:, 1]
    inside = (
        (qx >= 0) & (qx <= width - 1)
        & (qy >= 0) & (qy <= height - 1)
    )
    grid = torch.stack(
        (2.0 * qx / (width - 1) - 1.0,
         2.0 * qy / (height - 1) - 1.0),
        dim=-1
    )

    grad_x, grad_y = compute_depth_gradients(depth_ref)
    gradients = F.grid_sample(
        torch.cat((grad_x, grad_y), dim=1),
        grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    depth_valid = (depth_ref > 0) & torch.isfinite(depth_ref)
    support_count = F.avg_pool2d(
        depth_valid.to(depth_ref.dtype), kernel_size=3,
        stride=1, padding=1, divisor_override=1
    )
    support = (support_count == 9).to(depth_ref.dtype)
    support_at_q = F.grid_sample(
        support, grid, mode='bilinear',
        padding_mode='zeros', align_corners=True
    )

    normal_valid = (support_at_q >= 1.0 - 1e-6) & inside.unsqueeze(1)
    grad_norm = torch.linalg.vector_norm(gradients, dim=1, keepdim=True)
    normals = gradients / grad_norm.clamp_min(eps_g)
    normals = normals * normal_valid.to(normals.dtype)

    return normals, inside


def sequence_loss(flow_preds, flow_gt, depth_map, depth_ref, flow_valid,
                  gamma=0.8, MAX_FLOW=400, alpha=5.0,
                  eps=1e-6, eps_g=1e-5):
    with torch.no_grad():
        edge_weight = compute_depth_gradient_weight(
            depth_map, alpha=alpha, eps=eps
        )[:, 0]
        normals, inside = compute_boundary_normals(
            depth_ref, flow_gt, eps_g=eps_g
        )

        magnitude = torch.linalg.vector_norm(flow_gt, dim=1)
        valid = (
            flow_valid[:, 0].bool() & inside
            & torch.isfinite(flow_gt).all(dim=1)
            & (magnitude < MAX_FLOW)
        )
        valid = valid.to(flow_gt.dtype)
        valid_count = valid.sum(dim=(1, 2))

    flow_loss = flow_gt.new_zeros(())
    n_predictions = len(flow_preds)

    for i, flow_pred in enumerate(flow_preds):
        error = flow_pred - flow_gt
        base_error = torch.linalg.vector_norm(error, dim=1)
        normal_error = (error * normals).sum(dim=1).abs()

        point_loss = base_error + (edge_weight - 1.0) * normal_error
        sample_loss = (point_loss * valid).sum(dim=(1, 2))
        sample_loss = sample_loss / (valid_count + eps)

        flow_loss = flow_loss + pow(
            gamma, n_predictions - i - 1
        ) * sample_loss.mean()

    epe = (base_error.detach() * valid).sum() / (valid.sum() + eps)
    metrics = {'epe': epe.item()}
    return flow_loss, metrics


def sequence_loss_single(flow_pred, flow_gt, depth_map, depth_ref, flow_valid,
                         MAX_FLOW=400, alpha=5.0, eps=1e-6, eps_g=1e-5):
    flow_loss, _ = sequence_loss(
        [flow_pred], flow_gt, depth_map, depth_ref, flow_valid,
        MAX_FLOW=MAX_FLOW, alpha=alpha, eps=eps, eps_g=eps_g
    )
    return flow_loss
