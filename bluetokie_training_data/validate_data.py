#!/usr/bin/env python3
"""
HISYNC AI - Data Validation Script
Validates collected images for training quality
"""

import os
from PIL import Image
from pathlib import Path

def validate_images(directory):
    """Validate images in directory"""
    valid_count = 0
    invalid_count = 0
    
    for img_path in Path(directory).rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                if img.size[0] >= 224 and img.size[1] >= 224:
                    valid_count += 1
                else:
                    print(f"Too small: {img_path}")
                    invalid_count += 1
        except Exception as e:
            print(f"Invalid: {img_path} - {e}")
            invalid_count += 1
    
    print(f"Valid: {valid_count}, Invalid: {invalid_count}")
    return valid_count, invalid_count

if __name__ == "__main__":
    validate_images("bluetokie_training_data")
