import os
import sys
import time 

import numpy as np
import argparse
import random
import torch

from core.datasets_m3ed import DatasetM3ED as Dataset
from core.backbone import Backbone_Event
from core.utils import (count_parameters, merge_inputs, fetch_optimizer, Logger)
from core.data_preprocess import Data_preprocess
from core.flow2pose import Flow2Pose, err_Pose
from core.losses import sequence_loss

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
import math

from core.flow_viz import flow_to_image
from core.quaternion_distances import quaternion_distance
from core.flow2pose import Flow2Pose, err_Pose
from core.data_preprocess import Data_preprocess

occlusion_kernel = 5
occlusion_threshold = 3
seed = 1234

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

def _init_fn(worker_id, seed):
    seed = seed
    print(f"Init worker {worker_id} with seed {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch):
    global occlusion_threshold, occlusion_kernel
    model.train()
    for i_batch, sample in enumerate(TrainImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']
    

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=600, w=960)
        
        # =============== Data cleaning  ===============
        # Force all NaN and Inf values resulting from Z->0 to be replaced with 0.0
        event_input = torch.nan_to_num(event_input, nan=0.0, posinf=0.0, neginf=0.0)
        lidar_input = torch.nan_to_num(lidar_input, nan=0.0, posinf=0.0, neginf=0.0)
        flow_gt = torch.nan_to_num(flow_gt, nan=0.0, posinf=0.0, neginf=0.0)
        # ==============================================================================
         
        optimizer.zero_grad()

        flow_preds = model(lidar_input, event_input, iters=args.iters)
        
        
        #loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
        # Pass in the depth map to calculate the depth perception loss
        loss, metrics = sequence_loss(flow_preds, flow_gt, depth_map=lidar_input, gamma=args.gamma, MAX_FLOW=400)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        logger.push(metrics) 


def test(args, TestImgLoader, model, device, cal_pose=False):
    global occlusion_threshold, occlusion_kernel
    model.eval()
    out_list, epe_list = [], []
    Time = 0.
    
    outliers, err_r_list, err_t_list = [], [], []
    init_err_r_list, init_err_t_list = [], []
    
    vis_dir = "./visualizations_628"
    os.makedirs(vis_dir, exist_ok=True)

    total_frames = len(TestImgLoader)

    # Flag: Ensure that the 3D Event Tensor graph is generated only once.
    generated_3d_plot = False

    # ==========================================================
    # Define Hook, and capture the internal parameters and characteristics of the SFT module.
    # ==========================================================
    activation = {}
    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook

    def get_sft_io(name_in, name_out):
        def hook(module, input, output):
            # SFT 的 input 是 tuple: (event_feat, depth_feat)
            activation[name_in] = input[0].detach() 
            activation[name_out] = output.detach()
        return hook

    hook_handles = []
    try:
        if hasattr(model.module.fnet_event, 'sft'):
            h1 = model.module.fnet_event.sft.conv_gamma.register_forward_hook(get_activation('gamma'))
            h2 = model.module.fnet_event.sft.conv_beta.register_forward_hook(get_activation('beta'))
            h3 = model.module.fnet_event.sft.register_forward_hook(get_sft_io('feat_before', 'feat_after'))
            hook_handles.extend([h1, h2, h3])
    except Exception as e:
        print(f"Hook registration failed: {e}")

    for i_batch, sample in enumerate(TestImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=600, w=960)

        # Data cleaning, eliminating NaN values
        event_input = torch.nan_to_num(event_input, nan=0.0, posinf=0.0, neginf=0.0)
        lidar_input = torch.nan_to_num(lidar_input, nan=0.0, posinf=0.0, neginf=0.0)
        flow_gt = torch.nan_to_num(flow_gt, nan=0.0, posinf=0.0, neginf=0.0)

        end = time.time()
        
        outputs = model(lidar_input, event_input, iters=24, test_mode=True)
        if len(outputs) == 2:
            _, flow_up = outputs
        else:
            _, flow_up, _, _ = outputs

        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        
        valid_gt = (flow_gt[:, 0, :, :] != 0) | (flow_gt[:, 1, :, :] != 0)
        val = valid_gt.view(-1) >= 0.5
        
        if val.sum() > 0:
            out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())

        if cal_pose:
            R_pred, T_pred, inliers, flag = Flow2Pose(flow_up, lidar_input, calib, MAX_DEPTH=args.max_depth, x=60, y=160, h=600, w=960)
            Time += time.time() - end
            
            init_R = R_err[0].unsqueeze(0).to(device)
            init_T = T_err[0].to(device)
            init_rot_error = quaternion_distance(init_R, torch.tensor([[1., 0., 0., 0.]]).to(device), device=device) * 180. / math.pi
            init_trans_error = torch.norm(init_T) * 100.
            init_err_r_list.append(init_rot_error.item())
            init_err_t_list.append(init_trans_error.item())
            
            if flag: # When failing, revert to the initial error.
                outliers.append(i_batch)
                err_r_list.append(init_rot_error.item())
                err_t_list.append(init_trans_error.item())
            else:
                if len(outputs) > 2: 
                    from core.utils_point import to_rotation_matrix, quaternion_from_matrix
                    RT_pred = to_rotation_matrix(R_pred, T_pred)
                    T_pred = RT_pred[:3, 3]
                    R_pred = quaternion_from_matrix(RT_pred)               
                err_r, err_t = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
                err_r_list.append(err_r.item())
                err_t_list.append(err_t.item())

            print(f"{i_batch:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.median(err_t_list):.5f} "
                  f"{np.median(err_r_list):.5f} {len(outliers)} {Time / (i_batch+1):.5f}")

        # ==========================================================
        # 1. Generate 3D Event Tensor graph dynamically (it is generated only once globally, with dpi=600)
        # ==========================================================
        if not generated_3d_plot and event_input.sum() > 0:
            fig_3d = plt.figure(figsize=(10, 8), facecolor='white')
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            # Use channel 0 of event_input (red indicates positive/ channel 1) and channel 1 (blue indicates negative/ channel 2)
            # To prevent system crash, perform reasonable downsampling on overly dense events.
            pos_idx = torch.nonzero(event_input[0, 0] > 0, as_tuple=False)
            if len(pos_idx) > 0:
                y_pos = pos_idx[:, 0].cpu().numpy()
                x_pos = pos_idx[:, 1].cpu().numpy()
                t_pos = event_input[0, 0, y_pos, x_pos].cpu().numpy()
                step = max(1, len(t_pos) // 8000) 
                ax_3d.scatter(x_pos[::step], t_pos[::step], y_pos[::step], c='red', s=2, alpha=0.6, label='Positive')

            if event_input.shape[1] > 1:
                neg_idx = torch.nonzero(event_input[0, 1] > 0, as_tuple=False)
                if len(neg_idx) > 0:
                    y_neg = neg_idx[:, 0].cpu().numpy()
                    x_neg = neg_idx[:, 1].cpu().numpy()
                    t_neg = event_input[0, 1, y_neg, x_neg].cpu().numpy()
                    step = max(1, len(t_neg) // 8000)
                    ax_3d.scatter(x_neg[::step], t_neg[::step], y_neg[::step], c='blue', s=2, alpha=0.6, label='Negative')

            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Time (t)')
            ax_3d.set_zlabel('Y')
            ax_3d.invert_zaxis() # Reverse the Y-axis to match the image coordinate system
            # Remove the title
            plt.tight_layout()
            #plt.savefig(os.path.join(vis_dir, "Event_Tensor_3D.png"), dpi=600, bbox_inches='tight')
            plt.close(fig_3d)
            
            generated_3d_plot = True

        # ==========================================================
        # 2. Obtain the features and perform morphological and normalization processing
        # ==========================================================
        gt_flow_np = flow_gt[0].cpu().numpy().transpose(1, 2, 0)
        pred_flow_np = flow_up[0].detach().cpu().numpy().transpose(1, 2, 0)
        
        gt_flow_img = flow_to_image(gt_flow_np)
        pred_flow_img = flow_to_image(pred_flow_np)
        
        kernel = np.ones((2, 2), np.uint8)
        gt_flow_img_dilated = cv2.erode(gt_flow_img, kernel, iterations=1)

        depth_img = lidar_input[0, 0].cpu().numpy()
        voxel_img = event_input[0].mean(dim=0).cpu().numpy()
        
        sft_gamma = robust_normalize(activation['gamma'][0].mean(dim=0)) if 'gamma' in activation else np.zeros_like(depth_img)
        sft_beta = robust_normalize(activation['beta'][0].mean(dim=0)) if 'beta' in activation else np.zeros_like(depth_img)
        feat_before = robust_normalize(activation['feat_before'][0].mean(dim=0)) if 'feat_before' in activation else np.zeros_like(depth_img)
        feat_after = robust_normalize(activation['feat_after'][0].mean(dim=0)) if 'feat_after' in activation else np.zeros_like(depth_img)

        # ==========================================================
        # 3. Save 8 pictures separately (without names or titles, save them with suffixes 1 to 8, dpi = 300)
        # ==========================================================
        imgs_to_save = [
            (depth_img, 'plasma'),           # 1: LiDAR Depth Map
            (voxel_img, 'gray'),             # 2: Event Density Map
            (sft_gamma, 'jet'),              # 3: Learned SFT Scale Mask
            (sft_beta, 'jet'),               # 4: Learned SFT Shift Mask
            (feat_before, 'jet'),            # 5: Event Feature (Before Modulation)
            (feat_after, 'jet'),             # 6: Event Feature (After Modulation)
            (gt_flow_img_dilated, None),     # 7: Ground Truth Flow
            (pred_flow_img, None)            # 8: Predicted Flow (Ours)
        ]

        for img_idx, (img_data, cmap) in enumerate(imgs_to_save, 1):
            fig, ax = plt.subplots(figsize=(6, 4))
            if cmap is not None:
                ax.imshow(img_data, cmap=cmap)
            else:
                ax.imshow(img_data)
            
            # Hide the axes and do not add any title
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            save_path = os.path.join(vis_dir, f"frame_{i_batch:05d}_{img_idx}.png")
            #plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    # Remove Hook
    for h in hook_handles:
        h.remove()

    # ==========================================================
    # 4. Draw a line chart of the globally compliant error distribution (separated into two independent graphs, without titles, dpi = 600)
    # ==========================================================
    if cal_pose and len(err_t_list) > 0:
        plot_err_r = np.minimum(np.array(err_r_list), np.array(init_err_r_list))
        plot_err_t = np.minimum(np.array(err_t_list), np.array(init_err_t_list))
        
        def moving_average(data, window_size=3):
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')
            
        plot_err_r_smooth = moving_average(plot_err_r)
        plot_err_t_smooth = moving_average(plot_err_t)
        indices = np.arange(len(err_r_list))

        # --- Rotation error figure ---
        fig_r, ax_r = plt.subplots(figsize=(10, 6), facecolor='white')
        ax_r.plot(indices, init_err_r_list, color='blue', linestyle='-', linewidth=1.2, label='Error of the initial poses')
        ax_r.plot(indices, plot_err_r_smooth, color='red', linestyle='--', linewidth=1.5, label='Error of the refined poses using ours')
        ax_r.set_xlabel('Index', fontsize=16)
        ax_r.set_ylabel('Rotation Error (deg)', fontsize=16)
        # remove title
        ax_r.yaxis.set_major_locator(ticker.MultipleLocator(1.0)) 
        ax_r.grid(True, linestyle=':', alpha=0.6)
        ax_r.legend(fontsize=14)
        plt.tight_layout()
        #plt.savefig(os.path.join(vis_dir, "Global_Rotation_Error_LineChart.png"), dpi=600, bbox_inches='tight')
        plt.close(fig_r)
        
        # --- Translation Error figure ---
        fig_t, ax_t = plt.subplots(figsize=(10, 6), facecolor='white')
        ax_t.plot(indices, init_err_t_list, color='blue', linestyle='-', linewidth=1.2, label='Error of the initial poses')
        ax_t.plot(indices, plot_err_t_smooth, color='red', linestyle='--', linewidth=1.5, label='Error of the refined poses using ours')
        ax_t.set_xlabel('Index', fontsize=16)
        ax_t.set_ylabel('Translation Error (cm)', fontsize=16)
        # remove title
        ax_t.yaxis.set_major_locator(ticker.MultipleLocator(10.0)) 
        ax_t.grid(True, linestyle=':', alpha=0.6)
        ax_t.legend(fontsize=14)
        plt.tight_layout()
        #plt.savefig(os.path.join(vis_dir, "Global_Translation_Error_LineChart.png"), dpi=600, bbox_inches='tight')
        plt.close(fig_t)

    if len(epe_list) == 0:
        return 0.0, 0.0

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe = np.median(epe_list)
    f1 = 100 * np.mean(out_list)
    
    if not cal_pose:
        return epe, f1
    else:
        return err_t_list, err_r_list, outliers, Time, epe, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        metavar='DIR',
                        default='/Datasets/Event_Datasets/M3ED/generated/Falcon',
                        help='path to dataset')
    parser.add_argument('--ev_input', 
                        '--event_representation',
                        type=str,
                        default='voxel5_pre_100000s')
    parser.add_argument('--test_sequence',
                        type=str,
                        default='falcon_indoor_flight_3')
    parser.add_argument('--load_checkpoints',
                        help="restore checkpoint")
    parser.add_argument('--epochs', 
                        default=150, 
                        type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--starting_epoch', 
                        default=0, 
                        type=int, 
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', 
                        default=2, 
                        type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', 
                        '--learning_rate', 
                        default=4e-5, 
                        type=float,
                        metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--wdecay', 
                        type=float, 
                        default=.00005)
    parser.add_argument('--epsilon', 
                        type=float, 
                        default=1e-8)
    parser.add_argument('--clip', 
                        type=float, 
                        default=1.0)
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.8, 
                        help='exponential weighting')
    parser.add_argument('--iters', 
                        type=int, 
                        default=12)
    parser.add_argument('--gpus', 
                        type=int, 
                        nargs='+', 
                        default=[0])
    parser.add_argument('--max_r', 
                        type=float, 
                        default=5.)
    parser.add_argument('--max_t', 
                        type=float, 
                        default=0.5)
    parser.add_argument('--max_depth', 
                        type=float, 
                        default=10.)
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=3)
    parser.add_argument('--mixed_precision', 
                        action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--evaluate_interval', 
                        default=1, 
                        type=int, 
                        metavar='N',
                        help='Evaluate every \'evaluate interval\' epochs ')
    parser.add_argument('-e', 
                        '--evaluate', 
                        dest='evaluate', 
                        action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()    

    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])

    batch_size = args.batch_size

    _init_fn(0, seed)

    model = torch.nn.DataParallel(Backbone_Event(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    if args.load_checkpoints is not None:
        model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)

    _init_fn(0, seed)

    def init_fn(x):
        return _init_fn(x, seed)

    dataset_test = Dataset(args.data_path,
                           event_representation=args.ev_input,
                           max_r=args.max_r, 
                           max_t=args.max_t,
                           split='test', 
                           test_sequence=args.test_sequence)
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    if args.evaluate:
        with torch.no_grad():
            err_t_list, err_r_list, outliers, Time, epe, f1 = test(args, TestImgLoader, model, device, cal_pose=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"epe {epe:.5f}  Mean {Time / len(TestImgLoader):.5f} per frame")
            print(f"Outliers number {len(outliers)}/{len(TestImgLoader)} {outliers}")
        sys.exit()

    dataset_train = Dataset(args.data_path,
                            event_representation=args.ev_input,
                            max_r=args.max_r, 
                            max_t=args.max_t,
                            split='train',
                            test_sequence=args.test_sequence)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=args.num_workers,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)
    print("Train length: ", len(TrainImgLoader))
    print("Test length: ", len(TestImgLoader))

    optimizer, scheduler = fetch_optimizer(args, len(TrainImgLoader), model)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, SUM_FREQ=100)

    datetime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    if not os.path.exists(f'./checkpoints/{datetime}'):
        os.mkdir(f'./checkpoints/{datetime}')

    starting_epoch = args.starting_epoch
    if starting_epoch > 0:
        for i in range(starting_epoch * len(TrainImgLoader)):
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        logger.total_steps = starting_epoch * len(TrainImgLoader)

    min_val_err = 9999.
    # Record the start time outside the loop
    start_time = time.time()
    for epoch in range(starting_epoch, args.epochs):
        
        # Statistical elapsed time
        elapsed_time = (time.time() - start_time) / 3600.0  
        print(f"\n[Current progress----] Epoch: {epoch+1}/{args.epochs} | It has been running: {elapsed_time:.2f} h")
        
        train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch)

        torch.cuda.empty_cache()

        if epoch % args.evaluate_interval == 0:
            epe, f1 = test(args, TestImgLoader, model, device)
            print("Validation M3ED: %f, %f" % (epe, f1))

            results = {'m3ed-epe': epe, 'm3ed-f1': f1}
            logger.write_dict(results)

            torch.save(model.state_dict(), f"./checkpoints/{datetime}/checkpoint.pth")

            if epe < min_val_err:
                min_val_err = epe
                torch.save(model.state_dict(), f'./checkpoints/{datetime}/best_model.pth')
             # Force clearing of memory fragmentation
            torch.cuda.empty_cache()