from setuptools import setup, find_packages

setup(
    name="foundationperception",
    version="0.2.0",
    packages=find_packages(exclude=["FoundationStereo", "FoundationPose", "sam3", "sam-3d-objects"]),
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "torch",
        "torchvision",
        "trimesh",
        "open3d",
        "omegaconf",
        "hydra-core",
        "pillow",
        "tqdm",
        "matplotlib",
        "scikit-learn",
    ],
    extras_require={
        "stereo": [
            "flash-attn",
        ],
        "pose": [
            "nvdiffrast",
        ],
        "all": [
            "flash-attn",
            "nvdiffrast",
        ],
    },
    python_requires=">=3.9",
    author="Youngsun Wi",
    author_email="ysunnysun56@gmail.com",
    description="A unified package for foundation model-based perception (stereo depth, pose estimation, segmentation, mesh generation)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MMintLab/foundationperception",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
