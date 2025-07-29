"""
HISYNC AI - Bluetokie Coffee Training Setup Guide
Complete guide for training with 10,000+ coffee and cafe images

© 2025 Hire Synchronisation Pvt. Ltd. All rights reserved.
Developed by: Abhishek Rajput (@abhi-hisync)
Client: Bluetokie - Coffee Bean Roaster Market Leader
"""

import os
import json
from pathlib import Path
import time

def create_training_structure():
    """Create the complete training directory structure for Bluetokie"""
    
    print("🔥 HISYNC AI - Bluetokie Coffee Training Setup")
    print("=" * 60)
    print("Client: Bluetokie Coffee Bean Roaster Market Leader")
    print("Target: 10,000+ coffee and cafe images for AI training")
    print("Developer: Abhishek Rajput (@abhi-hisync)")
    print("Company: Hire Synchronisation Pvt. Ltd.")
    print("=" * 60)
    
    # Create base directory structure
    base_dir = Path("bluetokie_training_data")
    base_dir.mkdir(exist_ok=True)
    
    # Coffee and cafe categories for Bluetokie
    categories = {
        # Core Coffee Products (Bluetokie Priority)
        'coffee_beans': {
            'description': 'Raw and roasted coffee beans, Arabica, Robusta, bean bags, sacks',
            'target_images': 800,
            'subcategories': ['arabica_beans', 'robusta_beans', 'roasted_beans', 'green_beans', 'coffee_sacks']
        },
        'espresso_machine': {
            'description': 'Commercial and home espresso machines, professional coffee equipment',
            'target_images': 700,
            'subcategories': ['commercial_espresso', 'home_espresso', 'steam_wand', 'group_head', 'portafilter']
        },
        'coffee_grinder': {
            'description': 'Burr grinders, blade grinders, commercial grinding equipment',
            'target_images': 400,
            'subcategories': ['burr_grinder', 'blade_grinder', 'commercial_grinder', 'hand_grinder']
        },
        'brewing_equipment': {
            'description': 'French press, pour over, V60, Chemex, AeroPress, brewing tools',
            'target_images': 600,
            'subcategories': ['french_press', 'pour_over', 'v60', 'chemex', 'aeropress', 'moka_pot']
        },
        
        # Coffee Beverages
        'espresso_drinks': {
            'description': 'Espresso shots, doppio, ristretto, espresso cups',
            'target_images': 500,
            'subcategories': ['espresso_shot', 'doppio', 'ristretto', 'lungo', 'espresso_cup']
        },
        'cappuccino': {
            'description': 'Cappuccino with milk foam, foam art, steamed milk',
            'target_images': 600,
            'subcategories': ['cappuccino_classic', 'foam_art', 'milk_foam', 'cappuccino_cup']
        },
        'latte': {
            'description': 'Latte with latte art, flat white, cortado, milk coffee',
            'target_images': 700,
            'subcategories': ['latte_art', 'flat_white', 'cortado', 'macchiato', 'latte_glass']
        },
        'cold_coffee': {
            'description': 'Iced coffee, cold brew, nitro coffee, frappuccino',
            'target_images': 500,
            'subcategories': ['cold_brew', 'iced_coffee', 'nitro_coffee', 'frappuccino', 'iced_latte']
        },
        
        # Cafe Environment
        'cafe_interior': {
            'description': 'Coffee shop interiors, seating areas, cafe atmosphere',
            'target_images': 800,
            'subcategories': ['seating_area', 'cafe_counter', 'interior_design', 'lighting', 'decoration']
        },
        'coffee_service': {
            'description': 'Menu boards, coffee service, barista stations, customer areas',
            'target_images': 600,
            'subcategories': ['menu_board', 'barista_station', 'service_counter', 'ordering_area']
        },
        'coffee_accessories': {
            'description': 'Coffee cups, mugs, saucers, takeaway cups, accessories',
            'target_images': 700,
            'subcategories': ['ceramic_cups', 'paper_cups', 'travel_mugs', 'saucers', 'coffee_sleeves']
        },
        
        # Food & Pastries
        'pastries_food': {
            'description': 'Croissants, muffins, pastries, cakes, coffee accompaniments',
            'target_images': 600,
            'subcategories': ['croissants', 'muffins', 'pastries', 'cakes', 'cookies', 'sandwiches']
        },
        
        # Professional & Commercial (Bluetokie Focus)
        'professional_equipment': {
            'description': 'Commercial roasters, industrial equipment, professional tools',
            'target_images': 500,
            'subcategories': ['coffee_roaster', 'commercial_equipment', 'industrial_grinder', 'roasting_facility']
        },
        'coffee_packaging': {
            'description': 'Coffee bags, branded packaging, Bluetokie products, retail packaging',
            'target_images': 600,
            'subcategories': ['coffee_bags', 'branded_packaging', 'retail_packaging', 'labels', 'bluetokie_products']
        },
        'barista_work': {
            'description': 'Baristas at work, coffee preparation, latte art creation, service',
            'target_images': 700,
            'subcategories': ['barista_preparing', 'latte_art_creation', 'milk_steaming', 'coffee_pouring', 'customer_service']
        }
    }
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    
    for category, info in categories.items():
        category_dir = base_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different data splits
        (category_dir / "raw").mkdir(exist_ok=True)
        (category_dir / "train").mkdir(exist_ok=True)
        (category_dir / "validation").mkdir(exist_ok=True)
        (category_dir / "test").mkdir(exist_ok=True)
        
        # Create subcategory directories
        for subcat in info['subcategories']:
            (category_dir / "raw" / subcat).mkdir(exist_ok=True)
        
        print(f"  ✅ {category}: Target {info['target_images']} images")
    
    # Create metadata file
    metadata = {
        "project_name": "HISYNC AI - Bluetokie Coffee Training Dataset",
        "client": "Bluetokie Coffee Bean Roaster",
        "total_categories": len(categories),
        "target_total_images": sum(info['target_images'] for info in categories.values()),
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "developer": "Abhishek Rajput (@abhi-hisync)",
        "company": "Hire Synchronisation Pvt. Ltd.",
        "categories": categories
    }
    
    with open(base_dir / "training_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Create detailed instructions
    create_detailed_instructions(base_dir, categories, metadata)
    
    print(f"\n🎉 Training structure created successfully!")
    print(f"📊 Total target images: {metadata['target_total_images']:,}")
    print(f"📁 Location: {base_dir.absolute()}")
    
    return base_dir, metadata

def create_detailed_instructions(base_dir, categories, metadata):
    """Create comprehensive training instructions"""
    
    instructions = f"""
# HISYNC AI - Bluetokie Coffee Training Guide

## 📊 Project Overview
- **Client**: Bluetokie Coffee Bean Roaster - Market Leader
- **Target**: {metadata['target_total_images']:,} high-quality coffee and cafe images
- **Categories**: {len(categories)} specialized coffee/cafe categories
- **Purpose**: Physical verification of cafes, restaurants, and coffee establishments

## 🎯 Training Data Categories

{chr(10).join([f"### {i+1}. {cat.replace('_', ' ').title()}" + chr(10) + f"**Target**: {info['target_images']} images" + chr(10) + f"**Description**: {info['description']}" + chr(10) + f"**Subcategories**: {', '.join(info['subcategories'])}" + chr(10) for i, (cat, info) in enumerate(categories.items())])}

## 📥 How to Collect 10,000+ Images

### Option 1: Manual Collection (High Quality)
1. **Professional Photography**: Take high-quality photos at Bluetokie locations
2. **Partner Cafes**: Collect images from Bluetokie partner establishments
3. **Equipment Suppliers**: Document coffee equipment and machinery
4. **Training Sessions**: Capture barista training and coffee preparation

### Option 2: Online Sources (Faster Collection)
```bash
# Use the automated downloader
python bluetokie_dataset_downloader.py
```

### Option 3: Public Datasets
- **Open Images Dataset**: Filter coffee/cafe related images
- **ImageNet**: Extract relevant coffee categories
- **Unsplash/Pexels**: Download using APIs
- **Coffee Industry Datasets**: Professional coffee databases

### Option 4: Hybrid Approach (Recommended)
1. Start with automated collection (5,000 images)
2. Add professional Bluetokie photos (3,000 images)
3. Include partner cafe documentation (2,000 images)

## 🔧 Data Organization Process

### Step 1: Collect Raw Images
Place all images in the respective `raw/` directories:
```
bluetokie_training_data/
├── coffee_beans/raw/arabica_beans/
├── espresso_machine/raw/commercial_espresso/
└── ...
```

### Step 2: Quality Control
- **Resolution**: Minimum 224x224 pixels
- **Format**: JPEG or PNG preferred
- **Quality**: Clear, well-lit, focused images
- **Variety**: Different angles, lighting, backgrounds
- **Authenticity**: Real coffee/cafe environments

### Step 3: Data Splitting
Run the data splitter to create train/validation/test splits:
```bash
python bluetokie_dataset_trainer.py
```

### Step 4: Training
Execute the training pipeline:
```bash
python train_bluetokie_model.py
```

## 📋 Image Collection Guidelines

### Coffee Beans (800 images)
- Different roast levels (light, medium, dark)
- Various bean origins (Arabica, Robusta)
- Different packaging (bags, bulk, samples)
- Close-up and bulk shots

### Espresso Machines (700 images)
- Commercial grade equipment
- Home espresso machines
- Different angles and settings
- In-use and idle states

### Beverages (1,800 images total)
- Professional presentation
- Various cup types and sizes
- Different lighting conditions
- Before and after consumption

### Cafe Environment (1,400 images total)
- Interior design variations
- Different seating arrangements
- Menu displays and signage
- Customer interaction areas

## 🚀 Training Pipeline

### Phase 1: Data Preparation
1. Image collection and organization
2. Quality validation and filtering
3. Data augmentation setup
4. Train/validation/test splitting

### Phase 2: Model Training
1. Base model initialization (MobileNetV2)
2. Transfer learning with coffee data
3. Fine-tuning for Bluetokie specifics
4. Validation and testing

### Phase 3: Deployment
1. Model optimization and compression
2. API integration testing
3. Bluetokie verification validation
4. Production deployment

## 💡 Pro Tips for 10,000+ Images

### Automated Collection
- Use multiple data sources simultaneously
- Implement quality filters during download
- Create diverse search terms for each category
- Schedule downloads during off-peak hours

### Manual Collection
- Partner with coffee shops for photo sessions
- Document Bluetokie events and training sessions
- Capture seasonal variations and special events
- Include diverse demographics and settings

### Quality Assurance
- Implement automated quality checks
- Use crowd-sourcing for validation
- Regular audits of collected data
- Continuous feedback loop

## 📈 Success Metrics

### Data Quality
- **Accuracy**: >95% correctly labeled images
- **Diversity**: Multiple angles, lighting, settings
- **Coverage**: All subcategories represented
- **Resolution**: High-quality, clear images

### Model Performance
- **Training Accuracy**: >90%
- **Validation Accuracy**: >85%
- **Coffee Relevance**: >95% for coffee items
- **Bluetokie Compliance**: Professional verification standards

## 🔧 Technical Requirements

### Hardware
- **Storage**: 50GB+ for raw images
- **RAM**: 16GB+ for training
- **GPU**: NVIDIA GPU recommended for training
- **CPU**: Multi-core processor for data processing

### Software
- Python 3.8+
- TensorFlow 2.x
- PIL/Pillow for image processing
- FastAPI for deployment

## 📞 Support

**Developer**: Abhishek Rajput (@abhi-hisync)  
**Email**: support@hisync.in  
**Company**: Hire Synchronisation Pvt. Ltd.  
**Client**: Bluetokie Coffee Bean Roaster  

For technical support, training assistance, or custom requirements, contact our team.

---
© 2025 HISYNC Technologies - Revolutionizing Coffee Industry with AI
"""
    
    with open(base_dir / "TRAINING_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    # Create simple scripts
    create_helper_scripts(base_dir)

def create_helper_scripts(base_dir):
    """Create helper scripts for data management"""
    
    # Data validation script
    validation_script = '''#!/usr/bin/env python3
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
'''
    
    with open(base_dir / "validate_data.py", 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    # Data statistics script
    stats_script = '''#!/usr/bin/env python3
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
    
    print(f"\\nTotal: {total_images:,} images")
    print(f"Target: 10,000+ images")
    print(f"Progress: {(total_images/10000)*100:.1f}%")

if __name__ == "__main__":
    show_stats()
'''
    
    with open(base_dir / "show_stats.py", 'w', encoding='utf-8') as f:
        f.write(stats_script)

def main():
    """Main setup function"""
    base_dir, metadata = create_training_structure()
    
    print("\n🎯 Next Steps:")
    print("1. 📥 Start collecting images using the guidelines")
    print("2. 🔧 Run automated downloader: python bluetokie_dataset_downloader.py")
    print("3. 📊 Check progress: python show_stats.py")
    print("4. ✅ Validate data: python validate_data.py")
    print("5. 🚀 Start training: python bluetokie_dataset_trainer.py")
    
    print("\n📚 Documentation:")
    print(f"📖 Training Guide: {base_dir}/TRAINING_GUIDE.md")
    print(f"📄 Metadata: {base_dir}/training_metadata.json")
    
    print("\n💼 Ready for Bluetokie Coffee AI Training!")
    print("🔥 Target: 10,000+ images for professional coffee verification")

if __name__ == "__main__":
    main()
