"""
Image Utility Functions
========================

Comprehensive image loading, validation, and preprocessing utilities.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ExifTags

# Import config - handle case where config might not be available yet
try:
    from config import IMAGE_CONFIG
except ImportError:
    IMAGE_CONFIG = None

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageInfo:
    """Container for image metadata."""
    path: str
    width: int
    height: int
    channels: int
    format: str
    file_size_bytes: int
    bit_depth: int
    has_exif: bool
    exif_data: Optional[Dict[str, Any]] = None


def validate_image(
    image_path: Union[str, Path],
    check_corruption: bool = True
) -> Tuple[bool, str]:
    """
    Validate that an image file is readable and not corrupted.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the image file
    check_corruption : bool
        Whether to fully load image to check for corruption
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    image_path = Path(image_path)
    
    # Check file exists
    if not image_path.exists():
        return False, f"File not found: {image_path}"
    
    # Check file extension
    supported = IMAGE_CONFIG.supported_formats if IMAGE_CONFIG else \
        ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    if image_path.suffix.lower() not in supported:
        return False, f"Unsupported format: {image_path.suffix}"
    
    # Check file size
    max_size = (IMAGE_CONFIG.max_file_size_mb if IMAGE_CONFIG else 50) * 1024 * 1024
    file_size = image_path.stat().st_size
    
    if file_size > max_size:
        return False, f"File too large: {file_size / 1024 / 1024:.2f}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    # Check for corruption by attempting to load
    if check_corruption:
        try:
            with Image.open(image_path) as img:
                img.verify()
            
            # Re-open and load pixels to catch more corruption
            with Image.open(image_path) as img:
                img.load()
                
        except Exception as e:
            return False, f"Corrupted image: {str(e)}"
    
    return True, "Valid"


def load_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    color_mode: str = "RGB",
    as_array: bool = True
) -> Union[np.ndarray, Image.Image]:
    """
    Load an image with optional resizing and color conversion.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the image file
    target_size : Tuple[int, int], optional
        Target (width, height) for resizing
    color_mode : str
        Color mode: "RGB", "BGR", "GRAY", "L"
    as_array : bool
        Return as numpy array if True, PIL Image if False
        
    Returns
    -------
    np.ndarray or PIL.Image
        Loaded image
        
    Raises
    ------
    ValueError
        If image cannot be loaded
    """
    image_path = Path(image_path)
    
    # Validate first
    is_valid, error_msg = validate_image(image_path, check_corruption=False)
    if not is_valid:
        raise ValueError(error_msg)
    
    try:
        # Load with PIL for better format support
        img = Image.open(image_path)
        
        # Convert color mode
        if color_mode.upper() in ["RGB", "BGR"]:
            img = img.convert("RGB")
        elif color_mode.upper() in ["GRAY", "L"]:
            img = img.convert("L")
        
        # Resize if needed
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        if as_array:
            arr = np.array(img)
            
            # Convert RGB to BGR for OpenCV compatibility
            if color_mode.upper() == "BGR" and len(arr.shape) == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            return arr
        
        return img
        
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise ValueError(f"Failed to load image: {e}")


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C) in RGB format
    target_size : Tuple[int, int], optional
        Target (width, height)
    normalize : bool
        Apply ImageNet normalization
    mean : Tuple[float, ...]
        Mean values for normalization
    std : Tuple[float, ...]
        Std values for normalization
        
    Returns
    -------
    np.ndarray
        Preprocessed image (C, H, W) normalized
    """
    # Resize if needed
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Apply normalization
    if normalize:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = (image - mean) / std
    
    # Convert to (C, H, W) format for PyTorch
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    else:
        image = np.expand_dims(image, axis=0)
    
    return image


def save_image(
    image: Union[np.ndarray, Image.Image],
    output_path: Union[str, Path],
    quality: int = 95
) -> Path:
    """
    Save image to disk.
    
    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Image to save
    output_path : str or Path
        Output file path
    quality : int
        JPEG/WebP quality (1-100)
        
    Returns
    -------
    Path
        Path to saved image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        # Handle BGR to RGB conversion
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Save with appropriate settings
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        image.save(output_path, "JPEG", quality=quality, optimize=True)
    elif output_path.suffix.lower() == '.png':
        image.save(output_path, "PNG", optimize=True)
    elif output_path.suffix.lower() == '.webp':
        image.save(output_path, "WebP", quality=quality)
    else:
        image.save(output_path)
    
    logger.debug(f"Saved image to {output_path}")
    return output_path


def get_image_info(image_path: Union[str, Path]) -> ImageInfo:
    """
    Extract comprehensive metadata from an image.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the image file
        
    Returns
    -------
    ImageInfo
        Image metadata container
    """
    image_path = Path(image_path)
    
    with Image.open(image_path) as img:
        # Basic info
        width, height = img.size
        mode = img.mode
        format_name = img.format or "Unknown"
        
        # Determine channels
        channel_map = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4}
        channels = channel_map.get(mode, len(mode))
        
        # Bit depth
        bit_depth = 8  # Default
        if hasattr(img, 'bits'):
            bit_depth = img.bits
        
        # EXIF data
        exif_data = None
        has_exif = False
        
        try:
            exif = img._getexif()
            if exif:
                has_exif = True
                exif_data = {
                    ExifTags.TAGS.get(key, key): value
                    for key, value in exif.items()
                    if key in ExifTags.TAGS
                }
        except Exception:
            pass
    
    return ImageInfo(
        path=str(image_path),
        width=width,
        height=height,
        channels=channels,
        format=format_name,
        file_size_bytes=image_path.stat().st_size,
        bit_depth=bit_depth,
        has_exif=has_exif,
        exif_data=exif_data
    )


def create_image_grid(
    images: list,
    grid_size: Tuple[int, int],
    cell_size: Tuple[int, int] = (128, 128),
    padding: int = 2,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a grid visualization of multiple images.
    
    Parameters
    ----------
    images : list
        List of images (numpy arrays or paths)
    grid_size : Tuple[int, int]
        (rows, cols) of the grid
    cell_size : Tuple[int, int]
        Size of each cell (width, height)
    padding : int
        Padding between cells
    background_color : Tuple[int, int, int]
        Background color (RGB)
        
    Returns
    -------
    np.ndarray
        Grid image
    """
    rows, cols = grid_size
    cell_w, cell_h = cell_size
    
    # Calculate total grid size
    grid_h = rows * cell_h + (rows + 1) * padding
    grid_w = cols * cell_w + (cols + 1) * padding
    
    # Create background
    grid = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)
    
    # Place images
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        # Load if path
        if isinstance(img, (str, Path)):
            img = load_image(img, target_size=cell_size, color_mode="RGB")
        else:
            img = cv2.resize(img, cell_size)
        
        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Calculate position
        y = padding + row * (cell_h + padding)
        x = padding + col * (cell_w + padding)
        
        grid[y:y+cell_h, x:x+cell_w] = img
    
    return grid
