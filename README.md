# A Depth-Guided Feature Modulation Approach for Event-to-LiDAR Indoor Localization

This repository is the official implementation of the paper "A Depth-Guided Feature Modulation Approach for Event-to-LiDAR Indoor Localization" (Submitted to IEEE Sensors Journal).

The core code, pre-trained models and configuration files are currently being organized. This will be done within 7 days of paper submission.

# Overview

## 1. Abstract
  In this paper, we address the problem of Event-to-LiDAR indoor localization in GPS-denied or visually degraded indoor environments. Event cameras offer high temporal resolution and wide dynamic range, overcoming the limitations of conventional cameras. However, a fundamental modality gap exists: events respond to textures and geometric edges, whereas LiDAR depth maps encode only geometry. Textures on flat walls trigger abundant events but lack corresponding depth structures, causing spurious matches. Sparse signals in low-texture regions further increase matching ambiguity. Existing methods either treat the cross-modal matching process as a black box or rely on explicit edge detectors, incurring substantial computational overhead. To address this, we propose a depth-guided feature modulation approach. It converts geometric depth priors into affine parameters to adaptively modulate event features, inducing hierarchical differentiation at depth discontinuities and enhancing responses at geometric edges. This effectively suppresses textural interference and injects structural priors without additional edge decoders. Furthermore, a depth-gradient weighted loss function drives the optical flow network to focus on highly constrained edges and reduce matching ambiguities in low-texture areas, yielding reliable 2D-3D correspondences. The dense correspondences are subsequently processed by a Perspective-n-Point (PnP) solver to recover accurate 6-DoF camera poses. Experiments on indoor sequences from M3ED and MVSEC datasets demonstrate that our method surpasses state-of-the-art approaches in pose estimation accuracy, with fewer parameters and lower latency.

  ### 2. Graphical Abstract & Framework

<img width="2327" height="1611" alt="图片13" src="https://github.com/user-attachments/assets/a5289a51-6edb-46b2-8d35-564c688dac10" />

# Environment & Installation

1. We have verified the environment configuration under `Python 3.11`, `PyTorch 2.2.1`, and `CUDA 12.1`. Follow the step-by-step instructions below to set up your environment.

 ```bash
# 1. Upgrade pip and core tools
pip install --upgrade pip setuptools wheel

# 2. Install core libraries (Python 3.11 compatible)
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

# Dataset Preparation

Our method is evaluated on two public datasets. You can download them from the official sources:

M3ED Dataset: https://github.com/m3ed/m3ed
MVSEC Dataset: https://daniilidis-group.github.io/mvsec/

We will provide our pre-processing scripts and specific split files soon.

# Citation & Contact

He Deng (Denghe2019@163.com)
Yan Zhuang (zhuang@dlut.edu.cn)



