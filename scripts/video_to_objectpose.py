#!/usr/bin/env python3
"""
Extract object poses from video using language-prompted segmentation.

This script processes video files (or image directories) to estimate 6DoF object poses
using a text prompt to identify the target object. It combines:
- SAM3 for text-prompted segmentation
- SAM3D for mesh generation (if no mesh provided)
- FoundationStereo for depth estimation (from stereo infrared pairs)
- FoundationPose for 6DoF pose estimation

Depth can be obtained either from:
1. FoundationStereo (--foundationstereo): Computes depth from infrared stereo pairs
2. Pre-computed depth images (--depth_dir): Uses existing depth .npy files

Usage:
    # With FoundationStereo (infrared stereo -> depth)
    python video_to_objectpose.py \\
        --image_dir path/to/rgb_images \\
        --prompt "red cup" \\
        --foundationstereo \\
        --infra1_dir path/to/infra1 \\
        --infra2_dir path/to/infra2 \\
        --baseline 0.05 \\
        --output output_dir

    # With pre-computed depth images
    python video_to_objectpose.py \\
        --image_dir path/to/rgb_images \\
        --prompt "box" \\
        --depth_dir path/to/depth \\
        --mesh path/to/mesh.obj \\
        --output output_dir
    
    # From video with depth
    python video_to_objectpose.py \\
        --video path/to/video.mp4 \\
        --prompt "red cup" \\
        --depth_dir path/to/depth \\
        --output output_dir
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
import sys
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # RTX 4090
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # For nvdiffrast headless rendering

import argparse
import gc
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import tqdm

# Add submodule paths to PYTHONPATH
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR / "sam3"))
sys.path.insert(0, str(SCRIPT_DIR / "sam-3d-objects"))
sys.path.insert(0, str(SCRIPT_DIR / "FoundationPose"))
sys.path.insert(0, str(SCRIPT_DIR / "FoundationStereo"))

# Import from submodules after path setup
from foundationperception.stereo import StereoDepthProcessor
from foundationperception.utils import create_transformation_matrix


def get_gpu_devices() -> Tuple[torch.device, torch.device]:
    """Get available GPU devices."""
    if torch.cuda.device_count() >= 2:
        return torch.device('cuda:0'), torch.device('cuda:1')
    elif torch.cuda.device_count() == 1:
        return torch.device('cuda:0'), torch.device('cuda:0')
    else:
        raise RuntimeError("No CUDA devices available")


def apply_bfloat16_safety_patches():
    """Apply safety patches to prevent BFloat16 issues."""
    original_float = torch.Tensor.float
    
    def safe_float(self):
        if self.dtype == torch.bfloat16:
            return self.to(torch.float32)
        return original_float(self)
    
    torch.Tensor.float = safe_float


def load_video_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


def load_image_sequence(image_dir: str, extensions: List[str] = None) -> List[np.ndarray]:
    """Load images from a directory as a sequence."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_dir = Path(image_dir)
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    frames = [np.array(Image.open(f)) for f in image_files]
    return frames


def initialize_sam3(device: torch.device):
    """Initialize SAM3 model for segmentation."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    print(f"Loading SAM3 model on {device}...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 loaded successfully")
    
    return processor


def segment_with_sam3(processor, image: np.ndarray, text_prompt: str) -> np.ndarray:
    """Segment object in image using SAM3 with text prompt."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    mask = output["masks"].squeeze().detach().cpu().numpy()
    
    return mask


def generate_mesh_with_sam3d(
    image: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    focal_length: float,
    object_name: str = "object",
    device: str = "1",
) -> str:
    """Generate 3D mesh from image and mask using SAM3D in subprocess."""
    import subprocess
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image and mask
    image_path = output_dir / f'image_{object_name}.png'
    mask_path = output_dir / f'mask_{object_name}.png'
    output_mesh_path = output_dir / f'mesh_{object_name}.obj'
    
    Image.fromarray(image).save(image_path)
    mask_uint8 = (mask.astype(np.float32) * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(mask_path)
    
    # Run SAM3D in subprocess to avoid CUDA context conflicts
    sam3d_script = SCRIPT_DIR / "sam-3d-objects" / "demo.py"
    
    cmd = [
        sys.executable,
        str(sam3d_script),
        "--image_path", str(image_path),
        "--mask_path", str(mask_path),
        "--output_path", str(output_mesh_path),
    ]
    
    print("Running SAM3D mesh generation in subprocess...")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    # Disable HuggingFace offline mode
    for var in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        env.pop(var, None)
    
    result = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR / "sam-3d-objects"),
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(f"SAM3D stdout:\n{result.stdout}")
    if result.stderr:
        print(f"SAM3D stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(f"SAM3D subprocess failed with return code {result.returncode}")
    
    # Try to parse aligned mesh path from output
    try:
        json_str = result.stdout.split("===SAM3D_RESULT_JSON===")[1].strip().split('\n')[0]
        sam3d_result = json.loads(json_str)
        aligned_mesh_path = sam3d_result['aligned_mesh_path']
        
        # Copy aligned mesh to output path
        mesh = trimesh.load(aligned_mesh_path, force='mesh')
        mesh.export(str(output_mesh_path))
    except (IndexError, KeyError, json.JSONDecodeError):
        print("Using default output path")
    
    return str(output_mesh_path)


def initialize_foundation_pose(
    mesh_path: str,
    device: torch.device,
    min_n_views: int = 20,
    inplane_step: int = 90,
):
    """Initialize FoundationPose estimator."""
    import nvdiffrast.torch as dr
    from foundationpose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.fix_normals()
    
    # Initialize nvdiffrast
    print(f"Initializing nvdiffrast on {device}...")
    torch.cuda.set_device(device)
    with torch.cuda.device(device):
        glctx = dr.RasterizeCudaContext()
    print("nvdiffrast initialized successfully")
    
    # Set default tensor type
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    
    # Initialize FoundationPose
    print(f"Initializing FoundationPose on {device}...")
    with torch.cuda.amp.autocast(enabled=False):
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
        estimator.to_device(str(device))
        estimator.make_rotation_grid(min_n_views=min_n_views, inplane_step=inplane_step)
    
    print("FoundationPose initialized successfully")
    return estimator, mesh


def save_overlay_gif(
    rgb_images: List[np.ndarray],
    poses: List[np.ndarray],
    mesh: trimesh.Trimesh,
    intrinsic: np.ndarray,
    output_path: Path,
    fps: int = 10,
):
    """Save visualization GIF with mesh overlay."""
    from PIL import Image as PILImage
    import io
    
    frames = []
    for rgb, pose in zip(rgb_images, poses):
        # Project mesh vertices to image
        vertices_homo = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
        vertices_cam = (pose @ vertices_homo.T).T[:, :3]
        
        # Project to 2D
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        u = (vertices_cam[:, 0] * fx / vertices_cam[:, 2] + cx).astype(int)
        v = (vertices_cam[:, 1] * fy / vertices_cam[:, 2] + cy).astype(int)
        
        # Draw overlay
        overlay = rgb.copy()
        h, w = rgb.shape[:2]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (vertices_cam[:, 2] > 0)
        
        for ui, vi in zip(u[valid], v[valid]):
            cv2.circle(overlay, (ui, vi), 2, (0, 255, 0), -1)
        
        frames.append(PILImage.fromarray(overlay))
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0
    )
    print(f"Saved overlay GIF to {output_path}")


def process_video(
    frames: List[np.ndarray],
    text_prompt: str,
    intrinsic: np.ndarray,
    depth_images: List[np.ndarray] = None,
    infra1_images: List[np.ndarray] = None,
    infra2_images: List[np.ndarray] = None,
    stereo_processor: 'StereoDepthProcessor' = None,
    mesh_path: str = None,
    output_dir: Path = None,
    device_sam: torch.device = None,
    device_pose: torch.device = None,
) -> Tuple[List[np.ndarray], str]:
    """
    Process video frames to extract object poses.
    
    Args:
        frames: List of RGB frames
        text_prompt: Text description of object to track
        intrinsic: Camera intrinsic matrix (3x3)
        depth_images: Optional list of depth images (same length as frames)
        infra1_images: Optional list of left infrared images (for FoundationStereo)
        infra2_images: Optional list of right infrared images (for FoundationStereo)
        stereo_processor: Optional StereoDepthProcessor instance (for FoundationStereo)
        mesh_path: Optional path to object mesh (will generate if not provided)
        output_dir: Output directory for results
        device_sam: Device for SAM3 model
        device_pose: Device for FoundationPose model
        
    Returns:
        Tuple of (poses_list, mesh_path)
    """
    if device_sam is None or device_pose is None:
        device_pose, device_sam = get_gpu_devices()
    
    output_dir = Path(output_dir) if output_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply safety patches
    apply_bfloat16_safety_patches()
    
    # Initialize SAM3 for segmentation
    sam3_processor = initialize_sam3(device_sam)
    
    # Get first frame segmentation
    print(f"Segmenting '{text_prompt}' in first frame...")
    first_mask = segment_with_sam3(sam3_processor, frames[0], text_prompt)
    
    # Save first mask
    mask_path = output_dir / "first_mask.png"
    Image.fromarray((first_mask * 255).astype(np.uint8)).save(mask_path)
    print(f"Saved first frame mask to {mask_path}")
    
    # Generate mesh if not provided
    if mesh_path is None or not os.path.exists(mesh_path):
        print("No mesh provided, generating with SAM3D...")
        
        # Clean up SAM3 before running SAM3D
        del sam3_processor
        gc.collect()
        torch.cuda.empty_cache()
        
        mesh_path = generate_mesh_with_sam3d(
            image=frames[0],
            mask=first_mask,
            output_dir=output_dir / "mesh",
            focal_length=intrinsic[0, 0],
            object_name=text_prompt.replace(" ", "_"),
            device="1",  # Use second GPU for SAM3D
        )
        
        # Reinitialize SAM3 after SAM3D
        sam3_processor = initialize_sam3(device_sam)
    
    print(f"Using mesh: {mesh_path}")
    
    # Initialize FoundationPose
    estimator, mesh = initialize_foundation_pose(mesh_path, device_pose)
    
    # Process all frames
    poses = []
    masks = []
    
    for i, frame in enumerate(tqdm.tqdm(frames, desc="Processing frames")):
        # Get segmentation mask
        if i == 0:
            mask = first_mask
        else:
            mask = segment_with_sam3(sam3_processor, frame, text_prompt)
        
        masks.append(mask)
        
        # Get depth: FoundationStereo > pre-computed depth > error
        if stereo_processor is not None and infra1_images is not None and infra2_images is not None:
            # Use FoundationStereo to compute depth from infrared stereo pair
            infra1 = infra1_images[i]
            infra2 = infra2_images[i]
            
            # Convert to grayscale if needed
            if len(infra1.shape) == 3:
                infra1 = infra1[..., 0]
            if len(infra2.shape) == 3:
                infra2 = infra2[..., 0]
            
            depth, _ = stereo_processor.process_images(infra1, infra2, frame)
            depth = depth.astype(np.float32)
            
            # Save computed depth
            depth_output_dir = output_dir / "depth"
            depth_output_dir.mkdir(parents=True, exist_ok=True)
            np.save(depth_output_dir / f'{i:06d}.npy', depth)
        elif depth_images is not None:
            # Use pre-computed depth images
            depth = depth_images[i]
        else:
            raise ValueError(
                "No depth source provided. Either use --foundationstereo with "
                "--infra1_dir and --infra2_dir, or provide --depth_dir"
            )
        
        # Clear memory before pose estimation
        gc.collect()
        torch.cuda.empty_cache()
        
        # Estimate pose
        with torch.cuda.device(device_pose):
            with torch.cuda.amp.autocast(enabled=False):
                if i == 0:
                    # First frame - full registration
                    pose = estimator.register(
                        K=intrinsic,
                        rgb=frame,
                        depth=depth,
                        ob_mask=mask,
                        iteration=5
                    )
                else:
                    # Track from previous pose
                    pose = estimator.track_one(
                        K=intrinsic,
                        rgb=frame,
                        depth=depth,
                        iteration=5
                    )
        
        poses.append(pose)
    
    # Save results
    poses_array = np.array(poses)
    np.save(output_dir / "poses.npy", poses_array)
    print(f"Saved {len(poses)} poses to {output_dir / 'poses.npy'}")
    
    # Save visualization GIF
    save_overlay_gif(
        rgb_images=frames,
        poses=poses,
        mesh=mesh,
        intrinsic=intrinsic,
        output_path=output_dir / "poses_overlay.gif",
        fps=10,
    )
    
    return poses, mesh_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract object poses from video using language-prompted segmentation"
    )
    parser.add_argument(
        "--video", type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--image_dir", type=str,
        help="Path to directory containing image sequence"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing the object to track (e.g., 'red cup', 'box')"
    )
    parser.add_argument(
        "--mesh", type=str, default=None,
        help="Path to object mesh file (optional, will generate if not provided)"
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--intrinsic", type=str, default=None,
        help="Path to camera intrinsic matrix file (3x3 numpy array)"
    )
    parser.add_argument(
        "--focal_length", type=float, default=525.0,
        help="Camera focal length if intrinsic file not provided"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--depth_dir", type=str, default=None,
        help="Path to directory containing pre-computed depth images (.npy files)"
    )
    
    # FoundationStereo arguments
    parser.add_argument(
        "--foundationstereo", action="store_true",
        help="Use FoundationStereo to compute depth from infrared stereo pairs"
    )
    parser.add_argument(
        "--infra1_dir", type=str, default=None,
        help="Path to directory containing left infrared images (required if --foundationstereo)"
    )
    parser.add_argument(
        "--infra2_dir", type=str, default=None,
        help="Path to directory containing right infrared images (required if --foundationstereo)"
    )
    parser.add_argument(
        "--depth_intrinsic", type=str, default=None,
        help="Path to depth camera intrinsic matrix file (for FoundationStereo)"
    )
    parser.add_argument(
        "--extrinsics", type=str, default=None,
        help="Path to extrinsics file [tx,ty,tz,qx,qy,qz,qw] (for FoundationStereo)"
    )
    parser.add_argument(
        "--baseline", type=float, default=0.05,
        help="Stereo baseline in meters (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if args.video is None and args.image_dir is None:
        parser.error("Must provide either --video or --image_dir")
    
    # Validate depth source
    if not args.foundationstereo and args.depth_dir is None:
        parser.error("Must provide either --foundationstereo (with --infra1_dir, --infra2_dir) or --depth_dir")
    
    if args.foundationstereo:
        if args.infra1_dir is None or args.infra2_dir is None:
            parser.error("--foundationstereo requires --infra1_dir and --infra2_dir")
    
    # Load frames
    if args.video:
        print(f"Loading video: {args.video}")
        frames = load_video_frames(args.video, args.max_frames)
    else:
        print(f"Loading images from: {args.image_dir}")
        frames = load_image_sequence(args.image_dir)
        if args.max_frames:
            frames = frames[:args.max_frames]
    
    print(f"Loaded {len(frames)} frames")
    
    # Load or create intrinsic matrix
    if args.intrinsic and os.path.exists(args.intrinsic):
        intrinsic = np.loadtxt(args.intrinsic)
    else:
        # Create default intrinsic from focal length
        h, w = frames[0].shape[:2]
        intrinsic = np.array([
            [args.focal_length, 0, w / 2],
            [0, args.focal_length, h / 2],
            [0, 0, 1]
        ])
    
    print(f"Camera intrinsic:\n{intrinsic}")
    
    # Initialize variables for depth processing
    depth_images = None
    infra1_images = None
    infra2_images = None
    stereo_processor = None
    
    if args.foundationstereo:
        # Load infrared images for FoundationStereo
        print(f"Loading infrared images for FoundationStereo...")
        infra1_images = load_image_sequence(args.infra1_dir)
        infra2_images = load_image_sequence(args.infra2_dir)
        
        if args.max_frames:
            infra1_images = infra1_images[:args.max_frames]
            infra2_images = infra2_images[:args.max_frames]
        
        print(f"Loaded {len(infra1_images)} infra1 and {len(infra2_images)} infra2 images")
        
        # Validate frame counts match
        if len(infra1_images) != len(frames) or len(infra2_images) != len(frames):
            print(f"Warning: Frame count mismatch - RGB: {len(frames)}, infra1: {len(infra1_images)}, infra2: {len(infra2_images)}")
            min_frames = min(len(frames), len(infra1_images), len(infra2_images))
            frames = frames[:min_frames]
            infra1_images = infra1_images[:min_frames]
            infra2_images = infra2_images[:min_frames]
            print(f"Using {min_frames} frames")
        
        # Load depth intrinsic (default to color intrinsic, which is standard for RealSense)
        if args.depth_intrinsic and os.path.exists(args.depth_intrinsic):
            depth_intrinsic = np.loadtxt(args.depth_intrinsic)
        else:
            depth_intrinsic = intrinsic.copy()  # Use color intrinsic as default
        
        # Load extrinsics (default to identity if not provided)
        if args.extrinsics and os.path.exists(args.extrinsics):
            extrinsics_vec = np.loadtxt(args.extrinsics)
        else:
            print("Using default extrinsics [0,0,0,0,0,0,1] (provide --extrinsics for accuracy)")
            extrinsics_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Initialize FoundationStereo processor
        print("Initializing FoundationStereo processor...")
        stereo_processor = StereoDepthProcessor(
            color_intrinsic=intrinsic,
            depth_intrinsic=depth_intrinsic,
            extrinsics_vec=extrinsics_vec,
            baseline=args.baseline,
        )
        print("FoundationStereo initialized")
        
    elif args.depth_dir and os.path.exists(args.depth_dir):
        # Load pre-computed depth images
        print(f"Loading depth images from: {args.depth_dir}")
        depth_dir = Path(args.depth_dir)
        depth_files = sorted(depth_dir.glob("*.npy"))
        depth_images = [np.load(f) for f in depth_files[:len(frames)]]
        print(f"Loaded {len(depth_images)} depth images")
    
    # Process video
    output_dir = Path(args.output)
    poses, mesh_path = process_video(
        frames=frames,
        text_prompt=args.prompt,
        intrinsic=intrinsic,
        depth_images=depth_images,
        infra1_images=infra1_images,
        infra2_images=infra2_images,
        stereo_processor=stereo_processor,
        mesh_path=args.mesh,
        output_dir=output_dir,
    )
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Poses: {output_dir / 'poses.npy'}")
    print(f"  - Mesh: {mesh_path}")
    print(f"  - Visualization: {output_dir / 'poses_overlay.gif'}")


if __name__ == "__main__":
    main()

