# FoundationPerception

A unified Python package for foundation model-based 6DoF object pose estimation from video using language prompts.

## Features

- **Language-Prompted Pose Estimation**: Identify and track objects using natural language (e.g., "red cup", "cardboard box")
- **FoundationStereo**: High-quality depth estimation from stereo infrared image pairs
- **FoundationPose**: 6DoF object pose estimation and tracking from RGB-D images
- **SAM3**: Segment Anything Model 3 for text-prompted image segmentation
- **SAM3D**: Single-image 3D mesh generation (generates mesh if not provided)

## Installation

### Prerequisites

- CUDA 12.x compatible GPU (tested on RTX 4090)
- Conda environment with Python 3.9+

### Clone with Submodules

```bash
git clone --recursive -b foundationperception https://github.com/MMintLab/FoundationPerception.git
cd FoundationPerception

# Or if already cloned:
git checkout foundationperception
git submodule update --init --recursive
```

### Create Conda Environment

```bash
conda create -n foundationperception python=3.9
conda activate foundationperception

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Install core package
pip install -e .
```

### Install Submodules

#### FoundationStereo
```bash
cd FoundationStereo
pip install -e .
# Download pretrained model to pretrained_models/
cd ..
```

#### FoundationPose
```bash
cd FoundationPose
pip install -r requirements.txt
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
cd ..
```

## Quick Start

### Video to Object Pose (CLI)

Extract 6DoF object poses from images using a text prompt:

**With FoundationStereo** (computes depth from infrared stereo pairs):
```bash
python scripts/video_to_objectpose.py \
    --image_dir path/to/rgb_images \
    --prompt "red cup" \
    --foundationstereo \
    --infra1_dir path/to/infra1 \
    --infra2_dir path/to/infra2 \
    --baseline 0.05 \
    --output output_dir
```

**With pre-computed depth**:
```bash
python scripts/video_to_objectpose.py \
    --image_dir path/to/rgb_images \
    --prompt "cardboard box" \
    --depth_dir path/to/depth \
    --output output_dir
```

**With existing mesh**:
```bash
python scripts/video_to_objectpose.py \
    --image_dir path/to/rgb_images \
    --prompt "object" \
    --depth_dir path/to/depth \
    --mesh path/to/mesh.obj \
    --output output_dir
```

### Output

The script generates:
- `poses.npy` - Array of 4x4 pose matrices for each frame
- `poses_overlay.gif` - Visualization with mesh overlay
- `first_mask.png` - Initial segmentation mask
- `depth/` - Computed depth maps (if using FoundationStereo)
- `mesh/` - Generated mesh (if not provided)

### Python API

```python
from foundationperception import StereoDepthProcessor

# Initialize stereo depth processor
processor = StereoDepthProcessor(
    color_intrinsic=K_color,
    depth_intrinsic=K_depth,
    extrinsics_vec=extrinsics,
    baseline=0.05
)

# Process stereo images
depth, pointcloud = processor.process_images(left_ir, right_ir, color_image)
```

## Project Structure

```
foundationperception/
├── foundationperception/       # Core Python package
│   ├── __init__.py
│   ├── stereo/                 # Stereo depth estimation
│   │   └── processor.py
│   └── utils.py
├── scripts/
│   └── video_to_objectpose.py  # Main CLI script
├── assets/                     # Example camera configs
├── FoundationStereo/           # Submodule
├── FoundationPose/             # Submodule
├── sam3/                       # Submodule
├── sam-3d-objects/             # Submodule
├── requirements.txt
├── setup.py
└── README.md
```

## ROS1 Support

For ROS1 integration with real-time depth estimation, see the `ros1` branch:
```bash
git checkout ros1
```

## License

MIT License. Submodules have their own licenses (NVIDIA, Meta).
