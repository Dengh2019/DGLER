import os
import h5py
import numpy as np
import torch
import cv2
import argparse
from utils import load_data, find_near_index
from tqdm import tqdm
from event_utils import events_to_voxel_timesync_torch

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/hy-tmp/datasets/M3ED",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="falcon_indoor_flight_1",
                    help="Sequence name for processing",
                    type=str)
    ap.add_argument("--camera",
                    default="left",
                    help="which camera to use",
                    type=str)
    ap.add_argument("--method",
                    default="ours_denoise",
                    help="Event representation method",
                    type=str)
    ap.add_argument("--time_window",
                    default=100000,
                    help="length of time window",
                    type=int)
    ap.add_argument("--save_dir",
                    default="/hy-tmp/datasets/M3ED_Processed",
                    help="Path to save preprocessed data",
                    type=str)
    args = ap.parse_args()

    root = args.dataset
    seq_dir = os.path.join(root, args.sequence)

    # Strictly match the exact name of the downloaded file
    h5_file = f"{args.sequence}_data.h5"
    pose_file = f"{args.sequence}_pose_gt.h5"

    data_path = os.path.join(seq_dir, h5_file)
    pose_path = os.path.join(seq_dir, pose_file)

    data = h5py.File(data_path,'r')
    event_data_ref = load_data(data, sensor='prophesee', camera='left')
    event_data = load_data(data, sensor='prophesee', camera=args.camera)
    poses = h5py.File(pose_path,'r')
    ts_map_prophesee_left = poses['ts_map_prophesee_left']

    out_file = os.path.join(args.save_dir, args.sequence, f"event_frames_{args.method}_pre_{args.time_window}", args.camera)
    
    # Recursive creation of directories
    os.makedirs(out_file, exist_ok=True)
    
    rows, cols = event_data['resolution'][1], event_data['resolution'][0]
    t_start = event_data['t'][0]

    # Extract the full set of data and do not perform any truncation.
    for i in tqdm(range(len(ts_map_prophesee_left)-2)):
        idx_cur = int(ts_map_prophesee_left[i+1])
        t_ref = event_data_ref['t'][idx_cur]

        idx_start, idx_cur, idx_end = find_near_index(event_data_ref['t'][idx_cur], event_data['t'], time_window=args.time_window*2)

        event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)

        # Download resume protection
        if os.path.exists(f"{out_file}/event_frame_{i:05d}.npy"):
            continue
        
        if args.method == "ours_denoise":
            r = 6
            B = 1
            R = 1
            threshold = 0.7
            total_range = np.arange(idx_start, idx_cur)
            subsequences = np.array_split(total_range, B)
            for subseq in subsequences:
                mask = np.zeros((rows, cols), dtype=bool)
                for idx in subseq:
                    y, x = event_data['y'][idx], event_data['x'][idx]
                    if event_data['p'][idx] > 0:
                        patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                        patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
                        patch[patch<0] = 0
                        event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                        event_time_image[y, x, 0] = event_data['t'][idx]
                    else:
                        patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                        patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
                        patch[patch<0] = 0
                        event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                        event_time_image[y, x, 1] = event_data['t'][idx]
                    patch = event_time_image[max(0, y-R):y+R+1, max(0, x-R):x+R+1, :]
                    valid_count = ((patch[:, :, 0] > 0) | (patch[:, :, 1] > 0)).sum()
                    if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
                        mask[y, x] = True
                    else:
                        mask[y, x] = False
                event_time_image[mask] *= 0
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
            event_time_image[event_time_image < 0] = 0
        else:
            raise "Method doesn't exit."

        now = np.array(event_time_image)
        np.save(f"{out_file}/event_frame_{i:05d}", now)