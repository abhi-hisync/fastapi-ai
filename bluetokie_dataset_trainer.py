"""
HISYNC AI - Coffee & Cafe Dataset Organizer and Trainer
Specialized for Bluetokie Coffee Bean Roaster and Restaurant/Cafe Auditing

© 2025 Hire Synchronisation Pvt. Ltd. All rights reserved.
Developed by: Abhishek Rajput (@abhi-hisync)
Client: Bluetokie - Coffee Bean Roaster Market Leader
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import logging
import time
import shutil
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BluetokieCoffeeDatasetOrganizer:
    """
    HISYNC AI - Coffee & Cafe Dataset Organizer
    Helps organize your coffee/cafe images for training
    """
    
    def __init__(self, base_path: str = "datasets/bluetokie_coffee"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Coffee and cafe categories for Bluetokie verification
        self.categories = {
            # Core Coffee Products (Bluetokie Focus)
            'coffee_beans': 'Raw and roasted coffee beans, coffee sacks, bean bags',
            'espresso_machine': 'Commercial and home espresso machines, coffee machines',
            'coffee_grinder': 'Burr grinders, blade grinders, commercial grinders',
            'coffee_brewing_equipment': 'French press, pour over, V60, Chemex, AeroPress',
            
            # Coffee Beverages
            'espresso': 'Espresso shots, doppio, ristretto in cups',
            'cappuccino': 'Cappuccino with foam art, milk foam designs',
            'latte': 'Latte with latte art, flat white, cortado',
            'americano': 'Black coffee, drip coffee, filter coffee',
            'cold_brew': 'Iced coffee, cold brew, nitro coffee',
            'specialty_drinks': 'Macchiato, mocha, frappuccino, specialty beverages',
            
            # Cafe Environment
            'cafe_interior': 'Coffee shop seating, cafe atmosphere, interior design',
            'coffee_counter': 'Barista station, coffee bar, service counter',
            'coffee_menu': 'Menu boards, price lists, digital displays',
            'coffee_packaging': 'Coffee bags, branded packaging, Bluetokie products',
            
            # Accessories & Supplies
            'coffee_cups': 'Ceramic mugs, paper cups, takeaway cups, travel mugs',
            'coffee_accessories': 'Filters, tampers, milk jugs, scales',
            'pastries_bakery': 'Croissants, muffins, pastries, coffee accompaniments',
            'coffee_tools': 'Measuring spoons, timers, temperature gauges',
            
            # Staff & Service
            'barista_working': 'Barista preparing coffee, latte art creation',
            'coffee_service': 'Serving coffee, customer interaction',
            'coffee_roasting': 'Roasting equipment, roasted beans cooling',
            'quality_control': 'Coffee cupping, quality testing, inspection'
        }
        
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create organized directory structure for coffee/cafe images"""
        logger.info("📁 Creating HISYNC AI directory structure for Bluetokie...")
        
        # Create main directories
        (self.base_path / "raw_images").mkdir(exist_ok=True)
        (self.base_path / "organized").mkdir(exist_ok=True)
        (self.base_path / "training").mkdir(exist_ok=True)
        (self.base_path / "validation").mkdir(exist_ok=True)
        (self.base_path / "test").mkdir(exist_ok=True)
        
        # Create category directories
        for category in self.categories.keys():
            (self.base_path / "organized" / category).mkdir(exist_ok=True)
            (self.base_path / "training" / category).mkdir(exist_ok=True)
            (self.base_path / "validation" / category).mkdir(exist_ok=True)
            (self.base_path / "test" / category).mkdir(exist_ok=True)
        
        # Create instructions file
        self.create_instructions_file()
        
        logger.info("✅ Directory structure created successfully!")
    
    def create_instructions_file(self):
        """Create instructions for organizing images"""
        instructions = f"""
# HISYNC AI - Bluetokie Coffee Dataset Instructions

## 📁 Directory Structure Created:

### Main Directories:
- `raw_images/` - Place all your coffee/cafe images here initially
- `organized/` - Images organized by category
- `training/` - Training set (70% of images)
- `validation/` - Validation set (20% of images)
- `test/` - Test set (10% of images)

### Categories for Coffee/Cafe Classification:

{chr(10).join([f"**{cat}**: {desc}" for cat, desc in self.categories.items()])}

## 🚀 How to Use:

1. **Collect Images**: Place all your coffee/cafe images in the `raw_images/` folder
2. **Organize**: Use the `organize_images()` function to sort images by category
3. **Split Dataset**: Use `split_dataset()` to create train/validation/test splits
4. **Train Model**: Run the training script to create your Bluetokie classifier

## 📊 Recommended Dataset Size:
- **Minimum**: 100 images per category (2,000 total)
- **Good**: 500 images per category (10,000 total)
- **Excellent**: 1,000+ images per category (20,000+ total)

## 🎯 Image Quality Guidelines:
- **Resolution**: Minimum 224x224 pixels
- **Format**: JPEG or PNG
- **Quality**: Clear, well-lit images
- **Variety**: Different angles, lighting, backgrounds
- **Authenticity**: Real cafe/restaurant environments

## 💼 For Bluetokie Verification:
Focus especially on:
- Coffee bean varieties and packaging
- Professional espresso equipment
- Authentic cafe environments
- Quality control processes
- Branded coffee products

---
© 2025 HISYNC Technologies - Developed by Abhishek Rajput
        """
        
        with open(self.base_path / "README.md", 'w') as f:
            f.write(instructions)
    
    def organize_images_interactive(self):
        """Interactive image organization helper"""
        raw_path = self.base_path / "raw_images"
        
        if not raw_path.exists() or not any(raw_path.iterdir()):
            logger.error("❌ No images found in raw_images/ directory")
            logger.info("📥 Please add your coffee/cafe images to the raw_images/ folder first")
            return
        
        logger.info("🔍 Starting interactive image organization for Bluetokie...")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(raw_path.glob(ext))
        
        if not image_files:
            logger.error("❌ No image files found in raw_images/")
            return
        
        logger.info(f"📊 Found {len(image_files)} images to organize")
        
        # Show categories
        print("\n🏷️  Available Categories:")
        for i, (category, description) in enumerate(self.categories.items(), 1):
            print(f"{i:2d}. {category}: {description}")
        print(f"{len(self.categories)+1:2d}. Skip this image")
        print(f"{len(self.categories)+2:2d}. Exit organizer")
        
        organized_count = 0
        
        for img_file in image_files:
            try:
                # Display image info
                print(f"\n📸 Image: {img_file.name}")
                
                # Get user input
                while True:
                    try:
                        choice = input("Select category (number): ").strip()
                        choice_num = int(choice)
                        
                        if choice_num == len(self.categories) + 2:  # Exit
                            logger.info("👋 Exiting organizer...")
                            return organized_count
                        elif choice_num == len(self.categories) + 1:  # Skip
                            break
                        elif 1 <= choice_num <= len(self.categories):
                            # Move to category
                            category = list(self.categories.keys())[choice_num - 1]
                            dest_path = self.base_path / "organized" / category / img_file.name
                            shutil.move(str(img_file), str(dest_path))
                            organized_count += 1
                            print(f"✅ Moved to {category}")
                            break
                        else:
                            print("❌ Invalid choice. Please try again.")
                    except ValueError:
                        print("❌ Please enter a number.")
                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
                        break
                        
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
        
        logger.info(f"🎉 Organization complete! {organized_count} images organized.")
        return organized_count
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split organized images into train/validation/test sets"""
        logger.info("📊 Splitting dataset for Bluetokie training...")
        
        organized_path = self.base_path / "organized"
        
        if not organized_path.exists():
            logger.error("❌ No organized images found. Please organize images first.")
            return
        
        total_moved = 0
        
        for category in self.categories.keys():
            category_path = organized_path / category
            
            if not category_path.exists():
                continue
            
            # Get all images in category
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(category_path.glob(ext))
            
            if not images:
                logger.warning(f"⚠️  No images found in category: {category}")
                continue
            
            # Shuffle images
            random.shuffle(images)
            
            # Calculate split sizes
            total_images = len(images)
            train_size = int(total_images * train_ratio)
            val_size = int(total_images * val_ratio)
            test_size = total_images - train_size - val_size
            
            # Split images
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]
            
            # Move images to respective directories
            for img_list, folder in [(train_images, "training"), 
                                   (val_images, "validation"), 
                                   (test_images, "test")]:
                dest_dir = self.base_path / folder / category
                for img in img_list:
                    dest_path = dest_dir / img.name
                    shutil.move(str(img), str(dest_path))
                    total_moved += 1
            
            logger.info(f"📂 {category}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        logger.info(f"✅ Dataset split complete! {total_moved} images organized.")
        return total_moved


class BluetokieCoffeeTrainer:
    """
    HISYNC AI - Coffee & Cafe Neural Network Trainer
    Optimized for Bluetokie coffee verification
    """
    
    def __init__(self, dataset_path: str = "datasets/bluetokie_coffee"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.class_names = []
        self.history = None
        
        # Training configuration
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
    
    def load_dataset(self):
        """Load training and validation datasets"""
        logger.info("📊 Loading Bluetokie Coffee Dataset...")
        
        # Create data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path / "training",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            self.dataset_path / "validation",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        
        logger.info(f"✅ Dataset loaded successfully!")
        logger.info(f"📊 Training samples: {train_generator.samples}")
        logger.info(f"📊 Validation samples: {val_generator.samples}")
        logger.info(f"📊 Classes: {len(self.class_names)}")
        logger.info(f"🏷️  Categories: {', '.join(self.class_names)}")
        
        return train_generator, val_generator
    
    def create_model(self, num_classes: int):
        """Create specialized Bluetokie coffee classification model"""
        logger.info("🤖 Creating HISYNC AI Bluetokie Coffee Model...")
        
        # Use MobileNetV2 as base (optimized for deployment)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Create specialized coffee/cafe classifier
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu', name='coffee_features'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', name='cafe_features'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax', name='bluetokie_classifier')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info("✅ HISYNC AI Bluetokie Coffee Model Created!")
        return model
    
    def train_model(self):
        """Train the Bluetokie coffee classification model"""
        logger.info("🚀 Starting HISYNC AI Training for Bluetokie...")
        
        # Load dataset
        train_gen, val_gen = self.load_dataset()
        
        if train_gen.samples == 0:
            logger.error("❌ No training images found! Please organize your dataset first.")
            return None
        
        # Create model
        self.model = self.create_model(num_classes=len(self.class_names))
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/hisync_bluetokie_coffee_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("🎉 Initial training complete!")
        
        # Fine-tune with unfrozen layers
        logger.info("🔧 Fine-tuning model...")
        
        # Unfreeze top layers
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Fine-tune
        fine_tune_history = self.model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ HISYNC AI Training Complete!")
        return self.model
    
    def save_model_for_deployment(self):
        """Save model and metadata for deployment"""
        if self.model is None:
            logger.error("❌ No trained model to save!")
            return
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "hisync_bluetokie_coffee_v1.h5"
        self.model.save(model_path)
        
        # Create metadata for deployment
        metadata = {
            "model_name": "HISYNC AI - Bluetokie Coffee & Cafe Classifier",
            "version": "1.0.0",
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "client": "Bluetokie Coffee Bean Roaster",
            "purpose": "Physical verification of cafes and restaurants",
            "categories": self.class_names,
            "num_classes": len(self.class_names),
            "input_shape": [224, 224, 3],
            "preprocessing": "Rescale to [0,1], resize to 224x224",
            "developer": "Abhishek Rajput (@abhi-hisync)",
            "company": "Hire Synchronisation Pvt. Ltd.",
            "deployment_notes": "Replace existing model in image_classifier.py"
        }
        
        metadata_path = models_dir / "hisync_bluetokie_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save class mapping
        class_mapping = {i: name for i, name in enumerate(self.class_names)}
        mapping_path = models_dir / "hisync_bluetokie_classes.json"
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"💾 Metadata saved: {metadata_path}")
        logger.info(f"💾 Class mapping saved: {mapping_path}")
        
        return model_path, metadata_path, mapping_path


def main():
    """Main interface for Bluetokie coffee dataset preparation and training"""
    print("🔥 HISYNC AI - Bluetokie Coffee & Cafe Training System")
    print("=" * 60)
    print("Client: Bluetokie Coffee Bean Roaster Market Leader")
    print("Purpose: Physical verification of cafes and restaurants")
    print("Target Dataset Size: 10,000+ images")
    print("Developer: Abhishek Rajput (@abhi-hisync)")
    print("Company: Hire Synchronisation Pvt. Ltd.")
    print("=" * 60)
    
    organizer = BluetokieCoffeeDatasetOrganizer()
    
    while True:
        print("\n🎯 What would you like to do?")
        print("1. 📁 Setup dataset directories")
        print("2. 🏷️  Organize images by category (Interactive)")
        print("3. 📊 Split dataset (train/validation/test)")
        print("4. 🚀 Train Bluetokie coffee model")
        print("5. 📋 Show dataset statistics")
        print("6. ❌ Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            organizer.create_directory_structure()
            print("\n✅ Directory structure created!")
            print("📥 Next: Add your coffee/cafe images to the 'raw_images' folder")
            
        elif choice == "2":
            count = organizer.organize_images_interactive()
            if count:
                print(f"✅ Organized {count} images successfully!")
            
        elif choice == "3":
            count = organizer.split_dataset()
            if count:
                print(f"✅ Split {count} images into train/validation/test sets!")
            
        elif choice == "4":
            trainer = BluetokieCoffeeTrainer()
            model = trainer.train_model()
            if model:
                trainer.save_model_for_deployment()
                print("🎉 Training complete! Model ready for deployment.")
            
        elif choice == "5":
            # Show dataset statistics
            dataset_path = Path("datasets/bluetokie_coffee")
            if dataset_path.exists():
                print("\n📊 Dataset Statistics:")
                for split in ["training", "validation", "test"]:
                    split_path = dataset_path / split
                    if split_path.exists():
                        total = 0
                        for cat_path in split_path.iterdir():
                            if cat_path.is_dir():
                                count = len(list(cat_path.glob("*.*")))
                                total += count
                                print(f"  {split}/{cat_path.name}: {count} images")
                        print(f"  {split} total: {total} images")
            else:
                print("❌ No dataset found. Please setup directories first.")
        
        elif choice == "6":
            print("👋 Thank you for using HISYNC AI!")
            print("💼 Ready to revolutionize Bluetokie's coffee verification!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
