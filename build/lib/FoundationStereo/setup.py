from setuptools import setup, find_packages

setup(
    name="FoundationStereo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A stereo depth estimation package using foundation models",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FoundationStereo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 