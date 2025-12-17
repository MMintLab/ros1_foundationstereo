"""
FoundationStereo - A stereo depth estimation package using foundation models
"""
__version__ = "0.1.0"

# Import core functionality
from .Utils import *  # Import Utils first since other modules depend on it
from .core.utils.utils import InputPadder
from .core.foundation_stereo import FoundationStereo 