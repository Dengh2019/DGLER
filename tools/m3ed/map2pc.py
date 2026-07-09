import os
import h5py
import open3d as o3
import torch
import argparse
from utils import load_map
import tqdm 

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
    ap.add_argument("--save_dir",
                    default="/hy-tmp/datasets/M3ED_Processed",
                    help="Path to save preprocessed data",
                    type=str)
    args = ap.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # [修复1] 增加子文件夹路径，让代码能找到 falcon_indoor_flight_1 这个目录
    seq_dir = os.path.join(args.dataset, args.sequence)

    # [恢复原貌] 坚决使用原作者带有 _data 和 _global 的后缀命名！
    data_path = os.path.join(seq_dir, args.sequence + "_data.h5")
    pose_path = os.path.join(seq_dir, args.sequence + "_pose_gt.h5")
    pc_path = os.path.join(seq_dir, args.sequence + "_global.pcd")

    # [修复2] 解决 FileNotFoundError，必须用 os.makedirs 才能创建多级父目录
    out_path = os.path.join(args.save_dir, args.sequence, "local_maps")
    os.makedirs(out_path, exist_ok=True)

    data = h5py.File(data_path,'r')
    prophesee_left_T_lidar = torch.tensor(data["/ouster/calib/T_to_prophesee_left"], device=device, dtype=torch.float32)

    # # pose load
    poses = h5py.File(pose_path,'r')
    Cn_T_C0 = poses['Cn_T_C0']                                                  
    Ln_T_L0 = poses['Ln_T_L0']                                                  
    pose_ts = poses['ts']                                                       
    ts_map_prophesee_left = poses['ts_map_prophesee_left']                      

    # # pc map load
    vox_map = load_map(pc_path, device)                                         
    print(f'load pointclouds finished! {vox_map.shape[1]} points')

    for idx in tqdm.tqdm(range(Ln_T_L0.shape[0]-1)):
        file = os.path.join(out_path, f'point_cloud_{idx:05d}.h5')
        if os.path.exists(file):
            continue
        pose = torch.tensor(Ln_T_L0[idx], device=device, dtype=torch.float32)
        local_map = vox_map.clone()
        local_map = torch.matmul(pose, local_map)
        indexes = local_map[0, :] > -1.
        
        ## falcon_indoor_flight
        indexes = indexes & (local_map[0, :] < 10.)
        indexes = indexes & (local_map[1, :] > -5.)
        indexes = indexes & (local_map[1, :] < 5.)

        local_map = local_map[:, indexes]
        local_map = torch.matmul(prophesee_left_T_lidar, local_map)

        with h5py.File(file, 'w') as hf:
            hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)