from setuptools import setup, find_packages

setup(
    name="mmint_foundationstereo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flash-attn",
        "scikit-learn",
        "scipy",
        "open3d",
        "trimesh",
        "matplotlib",
        "numpy",
        "pandas",
        
    ],
    python_requires=">=3.7",
    author="Youngsun Wi",
    author_email="ysunnysun56@gmail.com",
    description="A package for collecting and processing glove and robot data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MmintLab/ros1_foundationstereo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
