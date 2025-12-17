# FoundationPerception

A unified Python package for foundation model-based perception, combining state-of-the-art models for stereo depth estimation, 6DoF object pose estimation, image segmentation, and 3D mesh generation.

## Features

- **FoundationStereo**: High-quality stereo depth estimation from infrared image pairs
- **FoundationPose**: 6DoF object pose estimation from RGB-D images
- **SAM3**: Segment Anything Model 3 for text-prompted image segmentation
- **SAM3D**: Single-image 3D mesh generation (sam-3d-objects)

## Installation

### Prerequisites

- CUDA 12.x compatible GPU (tested on RTX 4090)
- Conda environment with Python 3.9+
- Git with submodule support

### Clone with Submodules

```bash
git clone --recursive https://github.com/MMintLab/foundationperception.git
cd foundationperception

# Or if already cloned:
git submodule update --init --recursive
```

### Create Conda Environment

```bash
# Create environment (recommended: use the foundationpose environment)
conda create -n foundationpose python=3.9
conda activate foundationpose

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -e .
```

### Install Submodules

Each submodule has its own installation requirements:

#### FoundationStereo
```bash
cd FoundationStereo
pip install -e .
# Download pretrained model
mkdir -p pretrained_models
# Place model_best_bp2.pth and cfg.yaml in pretrained_models/
cd ..
```

#### FoundationPose
```bash
cd FoundationPose
pip install -r requirements.txt
# Build CUDA extensions
bash build_all_conda.sh
cd ..
```

#### SAM3
```bash
cd sam3
pip install -e .
cd ..
```

#### SAM3D (sam-3d-objects)
```bash
cd sam-3d-objects
pip install -e .
# Download checkpoints
mkdir -p checkpoints/hf
# Follow instructions in sam-3d-objects/README.md for model downloads
cd ..
```

### Additional Dependencies

```bash
# For nvdiffrast (required by FoundationPose)
pip install nvdiffrast

# For flash attention (optional, improves performance)
pip install flash-attn --no-build-isolation
```

## Quick Start

### Stereo Depth Estimation

```python
import numpy as np
from foundationperception import StereoDepthProcessor

# Camera parameters
color_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
depth_intrinsic = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])
extrinsics_vec = [tx, ty, tz, qx, qy, qz, qw]  # depth-to-color transform
baseline = 0.05  # stereo baseline in meters

# Initialize processor
processor = StereoDepthProcessor(
    color_intrinsic=color_intrinsic,
    depth_intrinsic=depth_intrinsic,
    extrinsics_vec=extrinsics_vec,
    baseline=baseline
)

# Process stereo images
depth, pointcloud = processor.process_images(left_ir, right_ir, color_image)
```

### Video to Object Pose (CLI)

Extract 6DoF object poses from a video using a text prompt:

```bash
python scripts/video_to_objectpose.py \
    --video path/to/video.mp4 \
    --prompt "red cup" \
    --output output_dir \
    --focal_length 525.0
```

With pre-computed depth:
```bash
python scripts/video_to_objectpose.py \
    --image_dir path/to/rgb_images \
    --depth_dir path/to/depth_images \
    --prompt "cardboard box" \
    --mesh path/to/object.obj \
    --output output_dir
```

### Using Individual Components

#### SAM3 Segmentation
```python
import sys
sys.path.insert(0, "sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="red apple")
mask = output["masks"].squeeze().cpu().numpy()
```

#### FoundationPose
```python
import sys
sys.path.insert(0, "FoundationPose")
from foundationpose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
estimator = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx
)

# First frame: register
pose = estimator.register(K=intrinsic, rgb=rgb, depth=depth, ob_mask=mask)

# Subsequent frames: track
pose = estimator.track_one(K=intrinsic, rgb=rgb, depth=depth)
```

## ROS1 Support

This package also supports ROS1 integration for real-time depth estimation:

```bash
# Build the catkin workspace
cd ~/catkin_ws
catkin_make

# Launch RealSense cameras
roslaunch foundationperception launch_realsense.launch

# Run stereo depth estimation
rosrun foundationperception stereo.py
```

### Docker (Recommended for ROS1)

```bash
# Pull pre-built image
docker pull yswi0506/foundationstereo_multi:latest

# Or build from scratch
cd docker/scripts
./build_cuda12_ros1_multi.sh
```

## Project Structure

```
foundationperception/
├── foundationperception/       # Main Python package
│   ├── __init__.py
│   ├── stereo/                 # Stereo depth processing
│   │   ├── __init__.py
│   │   └── processor.py
│   └── utils.py               # Utility functions
├── scripts/                    # Executable scripts
│   ├── video_to_objectpose.py  # Video → pose estimation
│   ├── stereo.py              # ROS stereo node
│   └── ...
├── FoundationStereo/          # Submodule: stereo depth
├── FoundationPose/            # Submodule: pose estimation
├── sam3/                      # Submodule: segmentation
├── sam-3d-objects/            # Submodule: mesh generation
├── launch/                    # ROS launch files
├── assets/                    # Camera configs
└── docker/                    # Docker support
```

## Configuration

Camera parameters are stored in `assets/`:
- `assets/single/`: Single camera setup
- `assets/multi/`: Multi-camera setup
- `assets/config.py`: General configuration

## Citation

If you use this package, please cite the original papers:

```bibtex
@article{foundationstereo2024,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={...},
  year={2024}
}

@article{foundationpose2024,
  title={FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects},
  author={Wen, Bowen and others},
  year={2024}
}

@article{sam3,
  title={SAM 3: Segment Anything in 3D with Neural Radiance Fields},
  author={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License. Note that submodules have their own licenses:
- FoundationStereo: [Check submodule]
- FoundationPose: NVIDIA License
- SAM3: Meta License
- SAM3D: Meta License

## Acknowledgments

- [NVIDIA FoundationPose](https://github.com/NVlabs/FoundationPose)
- [FoundationStereo](https://github.com/MMintLab/FoundationStereo)
- [Meta SAM3](https://github.com/facebookresearch/sam3)
- [Meta SAM3D](https://github.com/facebookresearch/sam-3d-objects)
