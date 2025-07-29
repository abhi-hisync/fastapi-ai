"""
HISYNC AI - Bluetokie Coffee Dataset Builder
Automated collection of 10,000+ coffee and cafe images for training

© 2025 Hire Synchronisation Pvt. Ltd. All rights reserved.
Developed by: Abhishek Rajput (@abhi-hisync)
Client: Bluetokie - Coffee Bean Roaster Market Leader
"""

import os
import json
import logging
import time
import shutil
import requests
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urljoin
import concurrent.futures
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BluetokieDatasetDownloader:
    """
    HISYNC AI - Automated Coffee & Cafe Dataset Downloader
    Downloads 10,000+ high-quality images for Bluetokie training
    """
    
    def __init__(self, base_path: str = "datasets/bluetokie_10k"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.download_lock = Lock()
        self.stats = {"downloaded": 0, "failed": 0, "skipped": 0}
        
        # Coffee/Cafe search terms optimized for Bluetokie
        self.search_terms = {
            'coffee_beans': [
                'coffee beans roasted', 'arabica coffee beans', 'robusta coffee beans',
                'coffee bean bag', 'coffee sack', 'green coffee beans', 'coffee cherry',
                'coffee plantation', 'coffee harvest', 'coffee bean close up'
            ],
            'espresso_machine': [
                'commercial espresso machine', 'professional coffee machine', 'barista machine',
                'espresso maker', 'steam wand', 'portafilter', 'group head espresso',
                'cafe equipment', 'coffee shop machine', 'espresso brewing'
            ],
            'coffee_grinder': [
                'burr coffee grinder', 'commercial coffee grinder', 'espresso grinder',
                'coffee mill', 'blade grinder', 'coffee grinding', 'grind coffee beans'
            ],
            'coffee_brewing': [
                'french press coffee', 'pour over coffee', 'v60 coffee', 'chemex coffee',
                'aeropress coffee', 'coffee filter', 'coffee dripper', 'moka pot',
                'coffee brewing methods', 'manual coffee brewing'
            ],
            'espresso_drinks': [
                'espresso shot', 'espresso cup', 'doppio espresso', 'ristretto',
                'espresso crema', 'espresso glass', 'short black coffee'
            ],
            'cappuccino': [
                'cappuccino coffee', 'cappuccino foam', 'milk foam coffee', 'cappuccino art',
                'steamed milk coffee', 'cappuccino cup', 'coffee foam art'
            ],
            'latte': [
                'latte coffee', 'cafe latte', 'latte art', 'milk coffee latte',
                'flat white coffee', 'cortado coffee', 'macchiato coffee'
            ],
            'cold_coffee': [
                'iced coffee', 'cold brew coffee', 'nitro coffee', 'iced latte',
                'frappuccino', 'coffee ice', 'cold coffee drink'
            ],
            'cafe_interior': [
                'coffee shop interior', 'cafe seating', 'coffee bar design',
                'cafe counter', 'coffee shop atmosphere', 'cafe decoration',
                'coffee shop furniture', 'cafe lighting'
            ],
            'coffee_menu': [
                'coffee menu board', 'cafe menu', 'coffee price list',
                'chalkboard coffee menu', 'digital menu coffee', 'coffee shop menu'
            ],
            'coffee_cups': [
                'coffee cup ceramic', 'paper coffee cup', 'takeaway coffee cup',
                'coffee mug', 'travel coffee mug', 'disposable coffee cup',
                'coffee cup sleeve', 'coffee cup collection'
            ],
            'pastries_bakery': [
                'croissant coffee', 'coffee muffin', 'danish pastry', 'coffee cake',
                'biscotti coffee', 'scone coffee', 'donut coffee', 'bagel coffee',
                'pastry display cafe'
            ],
            'barista_work': [
                'barista making coffee', 'latte art creation', 'coffee preparation',
                'barista pouring milk', 'coffee making process', 'barista uniform',
                'professional barista', 'coffee service'
            ],
            'coffee_roasting': [
                'coffee roaster machine', 'coffee roasting process', 'roasted coffee beans',
                'coffee roastery', 'coffee roasting facility', 'coffee bean cooling',
                'industrial coffee roaster'
            ],
            'coffee_packaging': [
                'coffee bag design', 'coffee package', 'branded coffee bag',
                'coffee label', 'coffee branding', 'coffee product packaging',
                'retail coffee package'
            ]
        }
        
        # Free image sources (no API key required)
        self.image_sources = [
            'https://source.unsplash.com/800x600/?{query}',
            'https://picsum.photos/800/600?random={seed}',
        ]
    
    def download_image_from_url(self, url: str, filepath: Path, timeout: int = 30) -> bool:
        """Download image from URL with validation"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False
            
            # Download and save
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate file size (minimum 10KB)
            if filepath.stat().st_size < 10240:
                filepath.unlink()
                return False
            
            with self.download_lock:
                self.stats["downloaded"] += 1
                
            return True
            
        except Exception as e:
            logger.debug(f"Failed to download {url}: {str(e)}")
            with self.download_lock:
                self.stats["failed"] += 1
            return False
    
    def download_unsplash_images(self, query: str, count: int, output_dir: Path):
        """Download images from Unsplash Source (no API key needed)"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def download_single_image(i):
            # Create unique URL with timestamp to avoid caching
            seed = int(time.time() * 1000) + i + random.randint(1, 1000)
            url = f"https://source.unsplash.com/800x600/?{query}&sig={seed}"
            
            filename = f"{query.replace(' ', '_')}_{hashlib.md5(f'{query}_{i}_{seed}'.encode()).hexdigest()[:12]}.jpg"
            filepath = output_dir / filename
            
            if filepath.exists():
                with self.download_lock:
                    self.stats["skipped"] += 1
                return
            
            success = self.download_image_from_url(url, filepath)
            if success:
                logger.debug(f"Downloaded: {filename}")
            else:
                logger.debug(f"Failed: {filename}")
            
            time.sleep(random.uniform(0.5, 1.5))  # Rate limiting
        
        # Use threading for faster downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(download_single_image, range(count))
    
    def download_category_images(self, category: str, terms: List[str], images_per_term: int = 100):
        """Download images for a specific category"""
        category_dir = self.base_path / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📂 Downloading {category} images...")
        
        for term in terms:
            term_dir = category_dir / term.replace(' ', '_').replace('/', '_')
            
            # Check if we already have enough images
            existing_images = len(list(term_dir.glob("*.jpg"))) if term_dir.exists() else 0
            needed_images = max(0, images_per_term - existing_images)
            
            if needed_images == 0:
                logger.info(f"  ✅ {term}: Already have {existing_images} images")
                continue
            
            logger.info(f"  🔍 {term}: Downloading {needed_images} images...")
            
            # Download from Unsplash
            self.download_unsplash_images(term, needed_images, term_dir)
            
            # Brief pause between terms
            time.sleep(1)
    
    def create_full_dataset(self, images_per_term: int = 100):
        """Create the complete Bluetokie coffee dataset"""
        logger.info("🔥 Starting HISYNC AI Dataset Creation for Bluetokie")
        logger.info(f"Target: {images_per_term} images per search term")
        
        total_terms = sum(len(terms) for terms in self.search_terms.values())
        estimated_total = total_terms * images_per_term
        
        logger.info(f"📊 Estimated total images: {estimated_total}")
        logger.info(f"📊 Categories: {len(self.search_terms)}")
        logger.info(f"📊 Search terms: {total_terms}")
        
        start_time = time.time()
        
        for category, terms in self.search_terms.items():
            category_start = time.time()
            initial_stats = self.stats.copy()
            
            self.download_category_images(category, terms, images_per_term)
            
            category_time = time.time() - category_start
            category_downloaded = self.stats["downloaded"] - initial_stats["downloaded"]
            
            logger.info(f"📈 {category}: {category_downloaded} images in {category_time:.1f}s")
        
        total_time = time.time() - start_time
        
        logger.info("🎉 HISYNC AI Dataset Creation Complete!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  ✅ Downloaded: {self.stats['downloaded']}")
        logger.info(f"  ❌ Failed: {self.stats['failed']}")
        logger.info(f"  ⏭️  Skipped: {self.stats['skipped']}")
        logger.info(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"💼 Ready for Bluetokie Coffee Verification Training!")
        
        # Create dataset summary
        self.create_dataset_summary()
        
        return self.stats["downloaded"]
    
    def create_dataset_summary(self):
        """Create a summary of the downloaded dataset"""
        summary = {
            "dataset_name": "HISYNC AI - Bluetokie Coffee & Cafe Dataset",
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "client": "Bluetokie Coffee Bean Roaster",
            "purpose": "Physical verification of cafes and restaurants",
            "total_images": self.stats["downloaded"],
            "categories": {},
            "developer": "Abhishek Rajput (@abhi-hisync)",
            "company": "Hire Synchronisation Pvt. Ltd."
        }
        
        # Count images per category
        for category in self.search_terms.keys():
            category_dir = self.base_path / category
            if category_dir.exists():
                total_images = 0
                for subdir in category_dir.iterdir():
                    if subdir.is_dir():
                        count = len(list(subdir.glob("*.jpg")))
                        total_images += count
                summary["categories"][category] = total_images
        
        # Save summary
        with open(self.base_path / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create README
        readme_content = f"""
# HISYNC AI - Bluetokie Coffee & Cafe Dataset

## 📊 Dataset Information
- **Total Images**: {self.stats['downloaded']:,}
- **Categories**: {len(self.search_terms)}
- **Created**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Client**: Bluetokie Coffee Bean Roaster
- **Purpose**: Physical verification of cafes and restaurants

## 🏷️ Categories
{chr(10).join([f"- **{cat}**: {summary['categories'].get(cat, 0)} images" for cat in self.search_terms.keys()])}

## 🚀 Next Steps
1. Run the training script: `python bluetokie_dataset_trainer.py`
2. Organize images into train/validation/test splits
3. Train the HISYNC AI model
4. Deploy for Bluetokie verification

## 🎯 Usage
This dataset is specifically curated for Bluetokie's coffee verification needs, including:
- Coffee bean identification
- Equipment verification
- Cafe environment assessment
- Quality control processes

---
© 2025 HISYNC Technologies - Developed by Abhishek Rajput (@abhi-hisync)
        """
        
        with open(self.base_path / "README.md", 'w') as f:
            f.write(readme_content)


def main():
    """Main interface for dataset creation"""
    print("🔥 HISYNC AI - Bluetokie Coffee Dataset Builder")
    print("=" * 60)
    print("Client: Bluetokie Coffee Bean Roaster Market Leader")
    print("Purpose: Physical verification of cafes and restaurants")
    print("Target: 10,000+ high-quality coffee & cafe images")
    print("Developer: Abhishek Rajput (@abhi-hisync)")
    print("Company: Hire Synchronisation Pvt. Ltd.")
    print("=" * 60)
    
    downloader = BluetokieDatasetDownloader()
    
    print(f"\n📂 Dataset will be created in: {downloader.base_path.absolute()}")
    print(f"🏷️ Categories: {len(downloader.search_terms)}")
    print(f"🔍 Search terms: {sum(len(terms) for terms in downloader.search_terms.values())}")
    
    while True:
        print("\n🎯 Options:")
        print("1. 🚀 Create full dataset (10,000+ images)")
        print("2. 📦 Create sample dataset (1,000 images)")
        print("3. 🔧 Custom dataset size")
        print("4. 📊 Show current dataset statistics")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Creating full Bluetokie dataset...")
            print("⏱️ This will take approximately 30-60 minutes")
            confirm = input("Continue? (y/n): ").lower().strip()
            if confirm == 'y':
                downloader.create_full_dataset(images_per_term=150)
        
        elif choice == "2":
            print("\n📦 Creating sample dataset...")
            downloader.create_full_dataset(images_per_term=20)
        
        elif choice == "3":
            try:
                images_per_term = int(input("Images per search term (1-500): "))
                if 1 <= images_per_term <= 500:
                    total_estimated = sum(len(terms) for terms in downloader.search_terms.values()) * images_per_term
                    print(f"📊 Estimated total images: {total_estimated}")
                    confirm = input("Continue? (y/n): ").lower().strip()
                    if confirm == 'y':
                        downloader.create_full_dataset(images_per_term=images_per_term)
                else:
                    print("❌ Please enter a number between 1 and 500")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "4":
            # Show statistics
            if downloader.base_path.exists():
                total_images = 0
                print("\n📊 Current Dataset Statistics:")
                for category in downloader.search_terms.keys():
                    category_dir = downloader.base_path / category
                    if category_dir.exists():
                        cat_total = 0
                        for subdir in category_dir.iterdir():
                            if subdir.is_dir():
                                count = len(list(subdir.glob("*.jpg")))
                                cat_total += count
                        total_images += cat_total
                        print(f"  {category}: {cat_total} images")
                print(f"\n📈 Total: {total_images} images")
            else:
                print("❌ No dataset found. Please create dataset first.")
        
        elif choice == "5":
            print("👋 Thank you for using HISYNC AI!")
            print("💼 Ready to revolutionize Bluetokie's coffee verification!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
