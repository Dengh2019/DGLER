# A Depth-Guided Feature Modulation Approach for Event-to-LiDAR Indoor Localization

This repository is the official implementation of the paper "A Depth-Guided Feature Modulation Approach for Event-to-LiDAR Indoor Localization" (Submitted to IEEE Sensors Journal).

# Overview

## 1. Abstract
  In this paper, we address the problem of Event-to-LiDAR indoor localization in GPS-denied or visually degraded indoor environments. Event cameras offer high temporal resolution and wide dynamic range, overcoming the limitations of conventional cameras. However, a fundamental modality gap exists: events respond to textures and geometric edges, whereas LiDAR depth maps encode only geometry. Textures on flat walls trigger abundant events but lack corresponding depth structures, causing spurious matches. Sparse signals in low-texture regions further increase matching ambiguity. Existing methods either treat the cross-modal matching process as a black box or rely on explicit edge detectors, incurring substantial computational overhead. To address this, we propose a depth-guided feature modulation approach. It converts geometric depth priors into affine parameters to adaptively modulate event features, inducing hierarchical differentiation at depth discontinuities and enhancing responses at geometric edges. This effectively suppresses textural interference and injects structural priors without additional edge decoders. Furthermore, a depth-gradient weighted loss function drives the optical flow network to focus on highly constrained edges and reduce matching ambiguities in low-texture areas, yielding reliable 2D-3D correspondences. The dense correspondences are subsequently processed by a Perspective-n-Point (PnP) solver to recover accurate 6-DoF camera poses. Experiments on indoor sequences from M3ED and MVSEC datasets demonstrate that our method surpasses state-of-the-art approaches in pose estimation accuracy, with fewer parameters and lower latency.

### 2. Graphical Abstract & Framework

<img width="2376" height="1646" alt="摘要图片加了白色底色的" src="https://github.com/user-attachments/assets/ec669369-263c-4bda-8749-a573a3f6b77e" />


# Environment & Installation

1. We have verified the environment configuration under `Python 3.11`, `PyTorch 2.2.1`, and `CUDA 12.1`. Follow the step-by-step instructions below to set up your environment.

 ```bash
# 1.Create and activate DGLER environment
conda create -n DGLER python=3.11 -y
conda activate DGLER

# 2. Upgrade pip and core tools
pip install --upgrade pip setuptools wheel

# 3. Install core libraries (Python 3.11 compatible)
pip install h5py==3.11.0 opencv-python==4.10.0.84 tqdm==4.66.4 matplotlib scikit-image pyyaml scipy open3d spconv-cu121
```
2. Our framework contains three custom C++ operator packages. You need to compile and install them locally:

```bash
# Compile blender-mathutils
cd blender-mathutils
python setup.py install
cd ..

# Compile core/correlation_package
cd core/correlation_package
python setup.py install
cd ../..

# Compile core/visibility_package
cd core/visibility_package
python setup.py install
cd ../..
```

3. Installing PoseLib

The download link for PoseLib is: https://github.com/PoseLib/PoseLib

```bash
# 1. Install system dependencies (Eigen3 is required)
sudo apt update && sudo apt install unzip libeigen3-dev -y

# 2. Unzip the PoseLib source code
unzip PoseLib-master.zip
cd PoseLib-master

# 3. Install build tools
pip install pybind11 pybind11-stubgen

# 4. Compile and install
python setup.py install
```

# Dataset & Data Preprocessing

Our method is evaluated on two public datasets. You can download them from the official sources:

M3ED Dataset: https://m3ed.io/data_overview

MVSEC Dataset: https://daniilidis-group.github.io/mvsec/




# Acknowledgement

This codebase references the excellent open-source projects **[EVLoc](https://github.com/EasonChen99/EVLoc)** (ICRA 2025) and **[LEAR](https://github.com/EasonChen99/LEAR)** (ICRA 2026). We sincerely thank the authors for their inspiring work and for making their code publicly available.

If you find this repository or our paper helpful, please also consider citing their works:

```bibtex
@inproceedings{chen2025evloc,
  title={EVLoc: Event-based Visual Localization in LiDAR Maps via Event-Depth Registration},
  author={Chen, Kuangyi and Zhang, Jun and Fraundorfer, Friedrich},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}

@article{chen2026lear,
  title={LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization},
  author={Chen, Kuangyi and Zhang, Jun and Hu, Y and Zhou, Y and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2603.01839},
  year={2026}
}
```

# Citation & Contact
If you have any questions or find our work helpful, please contact us at:

He Deng (Denghe2019@163.com)

Yan Zhuang (zhuang@dlut.edu.cn)



