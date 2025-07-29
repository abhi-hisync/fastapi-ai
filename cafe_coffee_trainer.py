"""
HISYNC AI - Coffee & Cafe Physical Verification Training System
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
import io
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import requests
from urllib.parse import urljoin
import hashlib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoffeeCafeDatasetCollector:
    """
    HISYNC AI - Automated Coffee & Cafe Dataset Collection
    Specialized for Bluetokie and restaurant/cafe physical verification
    """
    
    def __init__(self, base_path: str = "datasets/coffee_cafe"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Coffee and cafe categories for Bluetokie verification
        self.categories = {
            # Coffee Equipment & Beans (Bluetokie Focus)
            'coffee_beans': [
                'coffee beans', 'roasted coffee beans', 'arabica beans', 'robusta beans',
                'coffee bean bag', 'coffee sack', 'green coffee beans', 'coffee cherry'
            ],
            'espresso_machine': [
                'espresso machine', 'commercial espresso machine', 'coffee machine',
                'barista machine', 'steam wand', 'portafilter', 'group head'
            ],
            'coffee_grinder': [
                'coffee grinder', 'burr grinder', 'commercial coffee grinder',
                'coffee mill', 'espresso grinder', 'blade grinder'
            ],
            'coffee_brewing': [
                'french press', 'pour over', 'v60', 'chemex', 'aeropress',
                'coffee filter', 'coffee dripper', 'coffee pot', 'moka pot'
            ],
            
            # Coffee Products & Beverages
            'espresso': [
                'espresso shot', 'espresso cup', 'doppio', 'ristretto',
                'espresso crema', 'shot glass', 'demitasse'
            ],
            'cappuccino': [
                'cappuccino', 'cappuccino cup', 'milk foam', 'latte art',
                'steamed milk', 'coffee foam', 'cappuccino saucer'
            ],
            'latte': [
                'latte', 'cafe latte', 'latte art', 'milk coffee',
                'flat white', 'cortado', 'macchiato'
            ],
            'americano': [
                'americano', 'black coffee', 'drip coffee', 'filter coffee',
                'coffee mug', 'coffee cup'
            ],
            'cold_brew': [
                'cold brew', 'iced coffee', 'coffee ice', 'cold coffee',
                'nitro coffee', 'coffee tumbler', 'iced latte'
            ],
            
            # Cafe Environment & Equipment
            'cafe_interior': [
                'coffee shop interior', 'cafe seating', 'coffee bar',
                'cafe counter', 'coffee shop atmosphere', 'barista station'
            ],
            'coffee_menu': [
                'coffee menu', 'cafe menu board', 'coffee price list',
                'menu display', 'chalkboard menu', 'digital menu'
            ],
            'coffee_cups': [
                'coffee cup', 'disposable coffee cup', 'paper cup',
                'ceramic mug', 'travel mug', 'takeaway cup', 'coffee sleeve'
            ],
            'pastries': [
                'croissant', 'muffin', 'danish pastry', 'coffee cake',
                'biscotti', 'scone', 'donut', 'bagel'
            ],
            
            # Restaurant Verification Items
            'restaurant_coffee': [
                'restaurant coffee service', 'dinner coffee', 'after meal coffee',
                'coffee service tray', 'restaurant coffee cup'
            ],
            'coffee_packaging': [
                'coffee bag', 'coffee package', 'roasted coffee package',
                'coffee branding', 'coffee label', 'bluetokie package'
            ],
            'barista': [
                'barista', 'coffee preparation', 'latte art creation',
                'coffee pouring', 'steam milk', 'barista uniform'
            ],
            'coffee_roasting': [
                'coffee roaster', 'roasting machine', 'coffee roasting',
                'roasted beans cooling', 'coffee roastery', 'roasting facility'
            ]
        }
        
        # Image sources for dataset collection
        self.image_sources = {
            'unsplash_api': 'https://api.unsplash.com/search/photos',
            'pexels_api': 'https://api.pexels.com/v1/search'
        }
        
        # API keys (you'll need to get these)
        self.unsplash_key = os.getenv('UNSPLASH_ACCESS_KEY', '')
        self.pexels_key = os.getenv('PEXELS_API_KEY', '')
        
    def download_image(self, url: str, filepath: Path) -> bool:
        """Download image from URL with HISYNC AI optimization"""
        try:
            headers = {
                'User-Agent': 'HISYNC-AI-Dataset-Collector/1.0 (Bluetokie Coffee Verification)'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for consistency (but keep good quality)
            if img.size[0] > 1024 or img.size[1] > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Save image
            img.save(filepath, 'JPEG', quality=90)
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            return False
    
    def collect_unsplash_images(self, query: str, count: int = 100) -> List[str]:
        """Collect images from Unsplash API"""
        if not self.unsplash_key:
            logger.warning("Unsplash API key not provided")
            return []
        
        images = []
        per_page = min(30, count)  # Unsplash limit
        pages = (count + per_page - 1) // per_page
        
        headers = {'Authorization': f'Client-ID {self.unsplash_key}'}
        
        for page in range(1, pages + 1):
            try:
                params = {
                    'query': query,
                    'page': page,
                    'per_page': per_page,
                    'orientation': 'all'
                }
                
                response = requests.get(
                    self.image_sources['unsplash_api'],
                    headers=headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                for photo in data.get('results', []):
                    images.append(photo['urls']['regular'])
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Unsplash API error for query '{query}': {str(e)}")
                break
        
        return images[:count]
    
    def collect_pexels_images(self, query: str, count: int = 100) -> List[str]:
        """Collect images from Pexels API"""
        if not self.pexels_key:
            logger.warning("Pexels API key not provided")
            return []
        
        images = []
        per_page = min(80, count)  # Pexels limit
        pages = (count + per_page - 1) // per_page
        
        headers = {'Authorization': self.pexels_key}
        
        for page in range(1, pages + 1):
            try:
                params = {
                    'query': query,
                    'page': page,
                    'per_page': per_page
                }
                
                response = requests.get(
                    self.image_sources['pexels_api'],
                    headers=headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                for photo in data.get('photos', []):
                    images.append(photo['src']['large'])
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Pexels API error for query '{query}': {str(e)}")
                break
        
        return images[:count]
    
    def create_dataset(self, target_images_per_category: int = 500):
        """Create comprehensive coffee/cafe dataset for Bluetokie verification"""
        logger.info(f"🔥 Starting HISYNC AI dataset creation for Bluetokie Coffee Verification")
        logger.info(f"Target: {target_images_per_category} images per category")
        
        total_downloaded = 0
        
        for category, search_terms in self.categories.items():
            category_path = self.base_path / category
            category_path.mkdir(exist_ok=True)
            
            logger.info(f"📂 Processing category: {category}")
            
            category_downloaded = 0
            images_per_term = target_images_per_category // len(search_terms)
            
            for term in search_terms:
                term_path = category_path / term.replace(' ', '_').replace('/', '_')
                term_path.mkdir(exist_ok=True)
                
                logger.info(f"🔍 Searching for: {term}")
                
                # Collect from multiple sources
                unsplash_images = self.collect_unsplash_images(term, images_per_term // 2)
                pexels_images = self.collect_pexels_images(term, images_per_term // 2)
                
                all_images = unsplash_images + pexels_images
                random.shuffle(all_images)
                
                # Download images
                downloaded_count = 0
                for i, img_url in enumerate(all_images[:images_per_term]):
                    img_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
                    filename = f"{term.replace(' ', '_')}_{img_hash}.jpg"
                    filepath = term_path / filename
                    
                    if not filepath.exists():
                        if self.download_image(img_url, filepath):
                            downloaded_count += 1
                            category_downloaded += 1
                            total_downloaded += 1
                            
                            if downloaded_count % 10 == 0:
                                logger.info(f"  ✅ Downloaded {downloaded_count}/{images_per_term} for '{term}'")
                    
                    time.sleep(0.5)  # Be respectful to APIs
                
                logger.info(f"  📊 Term '{term}': {downloaded_count} images downloaded")
            
            logger.info(f"📈 Category '{category}': {category_downloaded} total images")
            
        logger.info(f"🎉 HISYNC AI Dataset Creation Complete!")
        logger.info(f"📊 Total images downloaded: {total_downloaded}")
        logger.info(f"💼 Ready for Bluetokie Coffee Verification Training")
        
        return total_downloaded


class CoffeeCafeTrainer:
    """
    HISYNC AI - Coffee & Cafe Specialized Neural Network Trainer
    Optimized for Bluetokie and restaurant/cafe physical verification
    """
    
    def __init__(self, dataset_path: str = "datasets/coffee_cafe"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.history = None
        
        # Training configuration for coffee/cafe verification
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess coffee/cafe dataset"""
        logger.info("🔥 Loading HISYNC AI Coffee/Cafe Dataset for Bluetokie...")
        
        images = []
        labels = []
        
        # Load images from each category
        for category_path in self.dataset_path.iterdir():
            if category_path.is_dir():
                category_name = category_path.name
                logger.info(f"📂 Loading category: {category_name}")
                
                category_images = 0
                
                # Load from subcategories
                for subcategory_path in category_path.iterdir():
                    if subcategory_path.is_dir():
                        for img_path in subcategory_path.glob("*.jpg"):
                            try:
                                # Load and preprocess image
                                img = Image.open(img_path)
                                img = img.convert('RGB')
                                img = img.resize(self.img_size)
                                
                                # Convert to array and normalize
                                img_array = np.array(img) / 255.0
                                
                                images.append(img_array)
                                labels.append(category_name)
                                category_images += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to load {img_path}: {str(e)}")
                
                logger.info(f"  ✅ Loaded {category_images} images from {category_name}")
        
        # Convert to numpy arrays
        X = np.array(images)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        y = tf.keras.utils.to_categorical(y_encoded)
        
        self.class_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"📊 Dataset Summary:")
        logger.info(f"  Total images: {len(X)}")
        logger.info(f"  Categories: {len(self.class_names)}")
        logger.info(f"  Image shape: {X.shape[1:]}")
        logger.info(f"  Categories: {', '.join(self.class_names)}")
        
        return X, y, labels
    
    def create_model(self, num_classes: int) -> tf.keras.Model:
        """Create specialized coffee/cafe classification model"""
        logger.info("🤖 Creating HISYNC AI Coffee/Cafe Neural Network...")
        
        # Use MobileNetV2 as base (optimized for mobile deployment)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head for coffee/cafe items
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu', name='coffee_features'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', name='cafe_features'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax', name='bluetokie_classifier')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info("✅ HISYNC AI Coffee/Cafe Model Created Successfully!")
        model.summary()
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """Train the coffee/cafe classification model"""
        logger.info("🚀 Starting HISYNC AI Training for Bluetokie Coffee Verification...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        logger.info(f"📊 Training set: {len(X_train)} images")
        logger.info(f"📊 Validation set: {len(X_val)} images")
        
        # Create model
        self.model = self.create_model(num_classes=len(self.class_names))
        
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
                'models/bluetokie_coffee_cafe_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation for better generalization
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("🎉 HISYNC AI Training Complete!")
        
        # Fine-tune with unfrozen layers
        logger.info("🔧 Fine-tuning HISYNC AI Model for Bluetokie...")
        
        # Unfreeze top layers of base model
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Fine-tune training
        fine_tune_history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=20,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ HISYNC AI Fine-tuning Complete!")
        
        return self.model
    
    def save_model_and_metadata(self):
        """Save trained model and metadata"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / "hisync_bluetokie_coffee_cafe_v1.h5"
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            "model_name": "HISYNC AI - Bluetokie Coffee & Cafe Classifier",
            "version": "1.0.0",
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "client": "Bluetokie Coffee Bean Roaster",
            "purpose": "Physical verification of cafes and restaurants",
            "classes": self.class_names,
            "num_classes": len(self.class_names),
            "input_shape": list(self.img_size) + [3],
            "developer": "Abhishek Rajput (@abhi-hisync)",
            "company": "Hire Synchronisation Pvt. Ltd."
        }
        
        metadata_path = models_dir / "hisync_bluetokie_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save label encoder
        import pickle
        encoder_path = models_dir / "hisync_bluetokie_label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"💾 Metadata saved: {metadata_path}")
        logger.info(f"💾 Label encoder saved: {encoder_path}")
        
        return model_path, metadata_path, encoder_path
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('HISYNC AI - Bluetokie Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('HISYNC AI - Bluetokie Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/hisync_bluetokie_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training pipeline for HISYNC AI Coffee/Cafe Classifier"""
    print("🔥 HISYNC AI - Bluetokie Coffee & Cafe Training System")
    print("=" * 60)
    print("Client: Bluetokie Coffee Bean Roaster")
    print("Purpose: Physical verification of cafes and restaurants")
    print("Developer: Abhishek Rajput (@abhi-hisync)")
    print("Company: Hire Synchronisation Pvt. Ltd.")
    print("=" * 60)
    
    # Step 1: Collect dataset
    collector = CoffeeCafeDatasetCollector()
    
    print("\n🤖 Would you like to collect new dataset? (y/n): ", end="")
    collect_data = input().lower().strip() == 'y'
    
    if collect_data:
        print("\n📊 How many images per category would you like? (default 500): ", end="")
        try:
            images_per_category = int(input() or "500")
        except ValueError:
            images_per_category = 500
        
        print(f"\n🚀 Starting dataset collection with {images_per_category} images per category...")
        print("⚠️  Note: You need Unsplash and Pexels API keys in environment variables:")
        print("   UNSPLASH_ACCESS_KEY and PEXELS_API_KEY")
        
        total_images = collector.create_dataset(images_per_category)
        print(f"\n✅ Dataset collection complete! Total images: {total_images}")
    
    # Step 2: Train model
    trainer = CoffeeCafeTrainer()
    
    print("\n🔥 Loading dataset and starting training...")
    X, y, labels = trainer.load_and_preprocess_data()
    
    print(f"\n📊 Dataset loaded: {len(X)} images, {len(trainer.class_names)} categories")
    print(f"Categories: {', '.join(trainer.class_names)}")
    
    print("\n🚀 Starting HISYNC AI training for Bluetokie...")
    model = trainer.train_model(X, y)
    
    # Step 3: Save model
    model_path, metadata_path, encoder_path = trainer.save_model_and_metadata()
    
    # Step 4: Plot results
    trainer.plot_training_history()
    
    print("\n🎉 HISYNC AI Training Complete!")
    print(f"📁 Model saved: {model_path}")
    print(f"📁 Metadata: {metadata_path}")
    print(f"📁 Label encoder: {encoder_path}")
    
    print("\n💼 Ready for Bluetokie Coffee Verification!")
    print("🔧 Next steps:")
    print("1. Update image_classifier.py to use the new model")
    print("2. Test with coffee/cafe images")
    print("3. Deploy for Bluetokie physical verification")


if __name__ == "__main__":
    main()
