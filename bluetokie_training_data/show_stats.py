#!/usr/bin/env python3
"""
HISYNC AI - Dataset Statistics
Shows current dataset statistics
"""

from pathlib import Path

def show_stats():
    """Show dataset statistics"""
    base_dir = Path("bluetokie_training_data")
    
    print("Dataset Statistics")
    print("=" * 50)
    
    total_images = 0
    
    for category_dir in base_dir.iterdir():
        if category_dir.is_dir() and category_dir.name != "__pycache__":
            raw_dir = category_dir / "raw"
            if raw_dir.exists():
                count = len(list(raw_dir.rglob("*.jpg"))) + len(list(raw_dir.rglob("*.png")))
                total_images += count
                print(f"{category_dir.name}: {count} images")
    
    print(f"\nTotal: {total_images:,} images")
    print(f"Target: 10,000+ images")
    print(f"Progress: {(total_images/10000)*100:.1f}%")

if __name__ == "__main__":
    show_stats()
