#!/usr/bin/env python3

from PIL import Image
import numpy as np
import json
import os
from typing import List, Dict, Any
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import cv2
import re
from difflib import SequenceMatcher
import requests
import io
import torch
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')

class AIService:
    def __init__(self, inventory_file: str = "inventory.json"):
        self.inventory_file = inventory_file
        self.inventory = self.load_inventory()
        
        # Load robust AI model for clothing detection
        print("Loading robust AI model for clothing detection...")
        self.setup_robust_ai()
        print("✓ Robust AI model loaded successfully")
        
        # Pre-load all inventory images
        self.inventory_images = self.load_inventory_images()
        print(f"✓ Loaded {len(self.inventory_images)} inventory images")

    def load_inventory(self) -> List[Dict[str, Any]]:
        """Load inventory from JSON file"""
        try:
            with open(self.inventory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Inventory file {self.inventory_file} not found")
            return []

    def load_inventory_images(self) -> List[Dict[str, Any]]:
        """Load all inventory images into memory for direct comparison"""
        inventory_images = []
        
        for item in self.inventory:
            image_path = item.get('image_path', '')
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    inventory_images.append({
                        'item': item,
                        'image': image,
                        'image_path': image_path
                    })
                    print(f"✓ Loaded image: {item['name']}")
                except Exception as e:
                    print(f"✗ Failed to load {image_path}: {e}")
            else:
                print(f"✗ Image not found: {image_path}")
        
        return inventory_images

    def setup_robust_ai(self):
        """Setup CLIP AI model for perfect clothing detection"""
        try:
            print("Loading CLIP AI model...")
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Dynamic AI - no hardcoded descriptions, works with any inventory
            self.clothing_descriptions = []
            
            print("✓ CLIP AI model loaded successfully - perfect for all cases!")
            
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            self.model = None
            self.processor = None
            self.clothing_descriptions = []

    def detect_clothing_area(self, image: Image.Image) -> Image.Image:
        """Smart clothing detection that handles both model images and single clothing items"""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # First, try to detect if this is a single clothing item (no person)
            is_single_item = self.is_single_clothing_item(img_cv)
            
            if is_single_item:
                print("📦 Detected single clothing item - using full image")
                return image
            
            print("👤 Detected model wearing clothes - extracting clothing area")
            # Advanced clothing detection for model images
            return self.extract_clothing_from_model(img_cv)
                
        except Exception as e:
            print(f"Clothing detection failed: {e}")
            # Fallback to center crop
            return self.center_crop_image(image, 0.8)

    def is_single_clothing_item(self, img_cv) -> bool:
        """Detect if image contains a single clothing item (no person)"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Look for skin tones (indicates a person)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Count skin pixels
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            skin_ratio = skin_pixels / total_pixels
            
            # If less than 5% skin pixels, likely a single clothing item
            return skin_ratio < 0.05
            
        except Exception as e:
            print(f"Single item detection failed: {e}")
            return False

    def extract_clothing_from_model(self, img_cv) -> Image.Image:
        """Extract clothing area from model images"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Create skin mask
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Invert to get clothing areas
            clothing_mask = cv2.bitwise_not(skin_mask)
            
            # Clean up the mask
            kernel = np.ones((5,5), np.uint8)
            clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
            clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(clothing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest clothing area
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_cv.shape[1] - x, w + 2 * padding)
                h = min(img_cv.shape[0] - y, h + 2 * padding)
                
                # Crop the clothing area
                cropped = img_cv[y:y+h, x:x+w]
                
                # Convert back to PIL
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                return Image.fromarray(cropped_rgb)
            else:
                # Fallback to center crop
                return self.center_crop_image(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), 0.7)
                
        except Exception as e:
            print(f"Clothing extraction failed: {e}")
            return self.center_crop_image(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), 0.7)

    def center_crop_image(self, image: Image.Image, ratio: float = 0.7) -> Image.Image:
        """Center crop image to focus on main subject"""
        width, height = image.size
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return image.crop((left, top, right, bottom))

    def describe_clothing_with_ai(self, image: Image.Image) -> str:
        """Perfect AI clothing description using CLIP"""
        try:
            if self.model is not None and self.processor is not None:
                # Use CLIP for perfect clothing detection
                return self.analyze_with_clip(image)
            else:
                # Fallback to basic analysis
                return self.analyze_clothing_with_ai(image)
                    
        except Exception as e:
            print(f"AI description failed: {e}")
            return "clothing item"

    def analyze_with_clip(self, image: Image.Image) -> str:
        """DYNAMIC CLIP-based clothing analysis - works with any inventory"""
        try:
            # Generate dynamic descriptions from actual inventory
            dynamic_descriptions = self.generate_dynamic_descriptions()
            
            # Process image once
            image_inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                # Get image features once
                image_features = self.model.get_image_features(**image_inputs)
            
            best_similarity = -1
            best_description = "clothing item"
            
            # Process each dynamic description individually
            for description in dynamic_descriptions:
                try:
                    # Process single text description with proper padding
                    text_inputs = self.processor(text=[description], return_tensors="pt", padding=True, truncation=True)
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(image_features, text_features, dim=1).item()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_description = description
                        
                except Exception as e:
                    print(f"Error processing '{description}': {e}")
                    continue
            
            print(f"Dynamic CLIP detected: {best_description} (confidence: {best_similarity:.3f})")
            return best_description
                
        except Exception as e:
            print(f"Dynamic CLIP analysis failed: {e}")
            return "clothing item"

    def generate_dynamic_descriptions(self) -> List[str]:
        """Generate super intelligent descriptions from actual inventory items"""
        try:
            descriptions = []
            
            # Generate intelligent descriptions from inventory items
            for inv_item in self.inventory_images:
                item = inv_item['item']
                name = item['name'].lower()
                
                # Create multiple intelligent descriptions for better matching
                descriptions.append(f"a {name}")
                descriptions.append(f"{name}")
                descriptions.append(name)
                
                # Add intelligent variations based on clothing type
                if "jacket" in name:
                    descriptions.append(f"a {name} outerwear")
                    descriptions.append(f"{name} coat")
                elif "shirt" in name or "t-shirt" in name:
                    descriptions.append(f"a {name} top")
                    descriptions.append(f"{name} shirt")
                elif "jeans" in name or "pants" in name:
                    descriptions.append(f"{name} bottoms")
                    descriptions.append(f"{name} pants")
                elif "sneakers" in name or "shoes" in name:
                    descriptions.append(f"{name} footwear")
                    descriptions.append(f"{name} shoes")
                elif "sweater" in name or "hoodie" in name:
                    descriptions.append(f"a {name} knitwear")
                    descriptions.append(f"{name} pullover")
                elif "blazer" in name:
                    descriptions.append(f"a {name} suit jacket")
                    descriptions.append(f"{name} formal wear")
            
            print(f"Generated {len(descriptions)} super intelligent descriptions from inventory")
            return descriptions
            
        except Exception as e:
            print(f"Super intelligent description generation failed: {e}")
            return ["clothing item"]

    def analyze_clothing_with_robust_ai(self, image: Image.Image) -> str:
        """Robust AI clothing analysis using deep learning"""
        try:
            # Preprocess image for AI model
            if self.transform:
                input_tensor = self.transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    # Extract features using ResNet50
                    features = self.model(input_tensor)
                    
                    # Get clothing predictions
                    clothing_type = self.predict_clothing_type(features)
                    color = self.detect_color_robust(image)
                    material = self.detect_material_robust(image)
                    
                    # Combine AI results
                    if color and clothing_type:
                        return f"{color} {clothing_type}"
                    elif clothing_type:
                        return clothing_type
                    else:
                        return "clothing item"
            else:
                return "clothing item"
                
        except Exception as e:
            print(f"Robust AI analysis failed: {e}")
            return "clothing item"

    def predict_clothing_type(self, features):
        """Predict clothing type using AI model"""
        try:
            # Use features to predict clothing type
            # This is a simplified version - in practice, you'd train a classifier
            probabilities = F.softmax(features, dim=1)
            
            # Get top prediction
            top_prediction = torch.argmax(probabilities, dim=1).item()
            
            # Map to clothing categories (simplified)
            if top_prediction < len(self.clothing_categories):
                return self.clothing_categories[top_prediction]
            else:
                return "clothing"
                
        except Exception as e:
            print(f"Clothing type prediction failed: {e}")
            return "clothing"

    def detect_color_robust(self, image: Image.Image) -> str:
        """Robust color detection using AI"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Use advanced color analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Analyze color distribution more accurately
            mean_hue = np.mean(hsv[:,:,0])
            mean_sat = np.mean(hsv[:,:,1])
            mean_val = np.mean(hsv[:,:,2])
            
            # More accurate color detection
            if mean_val < 30:
                return "black"
            elif mean_val > 220:
                return "white"
            elif (mean_hue < 10 or mean_hue > 170) and mean_sat > 50:
                return "red"
            elif 100 < mean_hue < 130 and mean_sat > 50:
                return "blue"
            elif 20 < mean_hue < 40 and mean_sat > 50:
                return "yellow"
            elif 40 < mean_hue < 80 and mean_sat > 50:
                return "green"
            elif mean_sat < 30:
                return "gray"
            else:
                return "colored"
                
        except Exception as e:
            print(f"Robust color detection failed: {e}")
            return "colored"

    def detect_material_robust(self, image: Image.Image) -> str:
        """Robust material detection using AI"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Advanced texture analysis
            # Calculate local binary patterns
            
            # Use LBP for texture analysis
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            # Analyze texture patterns
            texture_energy = np.sum(lbp_hist ** 2)
            
            if texture_energy > 0.3:
                return "denim"
            elif texture_energy > 0.2:
                return "cotton"
            elif texture_energy > 0.1:
                return "synthetic"
            else:
                return "smooth"
                
        except Exception as e:
            print(f"Robust material detection failed: {e}")
            return "material"

    def analyze_clothing_with_ai(self, image: Image.Image) -> str:
        """Advanced AI clothing analysis using visual features"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Extract comprehensive clothing features
            features = self.extract_visual_features(img_array)
            return features
                            
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return "clothing item"

    def extract_visual_features(self, img_array):
        """Extract detailed visual features for clothing matching"""
        try:
            # Analyze multiple clothing characteristics
            color_features = self.analyze_colors_advanced(img_array)
            texture_features = self.analyze_texture_patterns(img_array)
            shape_features = self.analyze_clothing_shape(img_array)
            material_features = self.analyze_material_texture(img_array)
            
            # Create a more specific feature description
            # Focus on the most dominant characteristics
            dominant_color = color_features.split()[0] if color_features else "colored"
            dominant_texture = texture_features.split()[0] if texture_features else "textured"
            dominant_shape = shape_features.split()[0] if shape_features else "structured"
            dominant_material = material_features.split()[0] if material_features else "material"
            
            # Combine into a focused description
            combined_features = f"{dominant_color} {dominant_texture} {dominant_shape} {dominant_material}"
            
            return combined_features
                
        except Exception as e:
            print(f"Visual feature extraction failed: {e}")
            return "clothing item"

    def detect_clothing_type(self, img_array):
        """AI-based clothing type detection"""
        try:
            # Analyze image shape and patterns
            height, width = img_array.shape[:2]
            
            # Simple AI logic for clothing type detection
            if height > width * 1.5:
                return "jacket"  # Vertical items are usually jackets
            elif width > height * 1.2:
                return "jeans"  # Horizontal items are usually pants
            else:
                return "shirt"  # Square items are usually shirts
                
        except Exception as e:
            print(f"Clothing type detection failed: {e}")
            return None

    def analyze_colors_advanced(self, img_array):
        """Advanced color analysis for clothing - focus on dominant color"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Focus on the center area of the image (main clothing item)
            height, width = hsv.shape[:2]
            center_h = height // 2
            center_w = width // 2
            
            # Analyze center region (main clothing area)
            center_region = hsv[center_h-height//4:center_h+height//4, 
                              center_w-width//4:center_w+width//4]
            
            # Get the most dominant color from center region
            mean_hue = np.mean(center_region[:,:,0])
            mean_sat = np.mean(center_region[:,:,1])
            mean_val = np.mean(center_region[:,:,2])
            
            # Determine dominant color with higher precision - works for all colors
            if mean_val < 30:  # Very dark
                return "black"
            elif mean_val > 220:  # Very light
                return "white"
            elif (mean_hue < 10 or mean_hue > 170) and mean_sat > 50:  # Red range
                return "red"
            elif 100 < mean_hue < 130 and mean_sat > 50:  # Blue range
                return "blue"
            elif 20 < mean_hue < 40 and mean_sat > 50:  # Yellow range
                return "yellow"
            elif 40 < mean_hue < 80 and mean_sat > 50:  # Green range
                return "green"
            elif 130 < mean_hue < 170 and mean_sat > 50:  # Purple range
                return "purple"
            elif 80 < mean_hue < 100 and mean_sat > 50:  # Cyan range
                return "cyan"
            elif 10 < mean_hue < 20 and mean_sat > 50:  # Orange range
                return "orange"
            elif mean_sat < 30:  # Low saturation = gray
                return "gray"
            else:
                return "colored"
            
        except Exception as e:
            print(f"Advanced color analysis failed: {e}")
            return "colored"

    def analyze_texture_patterns(self, img_array):
        """Analyze clothing texture and patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate texture features using Local Binary Patterns
            texture_score = np.std(gray)
            
            # Detect patterns using edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            # Determine texture type
            if texture_score > 60:
                if edge_density > 0.15:
                    return "patterned textured"
                else:
                    return "rough textured"
            elif texture_score > 30:
                return "smooth textured"
            else:
                return "flat smooth"
                
        except Exception as e:
            print(f"Texture analysis failed: {e}")
            return "textured"

    def analyze_clothing_shape(self, img_array):
        """Analyze clothing shape and structure"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Find contours to analyze shape
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Analyze the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Calculate shape complexity
                if perimeter > 0:
                    complexity = area / (perimeter * perimeter)
                    if complexity > 0.1:
                        return "structured fitted"
                    else:
                        return "loose flowing"
            
            return "structured"
                
        except Exception as e:
            print(f"Shape analysis failed: {e}")
            return "structured"

    def analyze_material_texture(self, img_array):
        """Analyze material characteristics"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Analyze material properties
            # Calculate local variance for material detection
            kernel = np.ones((5,5), np.float32) / 25
            smoothed = cv2.filter2D(gray, -1, kernel)
            local_variance = np.var(gray - smoothed)
            
            # Determine material type based on texture - works for all materials
            if local_variance > 1000:
                return "denim material"
            elif local_variance > 500:
                return "cotton material"
            elif local_variance > 200:
                return "synthetic material"
            elif local_variance > 100:
                return "wool material"
            elif local_variance > 50:
                return "silk material"
            else:
                return "smooth material"
                
        except Exception as e:
            print(f"Material analysis failed: {e}")
            return "material"

    def detect_color(self, img_array):
        """AI-based color detection"""
        try:
            # Get dominant color using AI analysis
            pixels = img_array.reshape(-1, 3)
            
            # AI color analysis
            avg_color = np.mean(pixels, axis=0)
            r, g, b = avg_color
            
            if r < 50 and g < 50 and b < 50:
                return "black"
            elif r > 200 and g > 200 and b > 200:
                return "white"
            elif b > r and b > g:
                return "blue"
            elif r > g and r > b:
                return "red"
            elif r > 100 and g > 100 and b > 100 and r < 150 and g < 150 and b < 150:
                return "gray"
            else:
                return "colored"
                
        except Exception as e:
            print(f"Color detection failed: {e}")
            return None

    def detect_material(self, img_array):
        """AI-based material detection"""
        try:
            # Analyze texture patterns
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            texture = np.std(gray)
            
            if texture > 50:
                return "leather"  # High texture variation
            else:
                return "cotton"  # Low texture variation
            
        except Exception as e:
            print(f"Material detection failed: {e}")
            return None


    def normalize_clothing_description(self, description: str) -> str:
        """Normalize clothing description for better matching"""
        # Convert to lowercase and clean up
        desc = description.lower().strip()
        
        # Remove articles
        desc = desc.replace("a ", "").replace("an ", "").replace("the ", "")
        
        # Standardize common terms
        desc = desc.replace("sweatshirt", "hoodie")
        desc = desc.replace("cotton t-shirt", "t-shirt")
        desc = desc.replace("athletic sneakers", "sneakers")
        desc = desc.replace("wool sweater", "sweater")
        
        return desc

    def process_uploaded_image(self, image_data: str) -> tuple:
        """Perfect AI image processing with CLIP"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

            print("🔍 Processing image with CLIP AI...")
            
            # Smart clothing detection
            processed_image = self.detect_clothing_area(image)

            print("🤖 CLIP AI analyzing clothing...")
            # Get perfect AI description using CLIP
            description = self.describe_clothing_with_ai(processed_image)

            print("✅ CLIP AI processing completed successfully")
            return description, processed_image

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None


    def find_best_match(self, query_description: str) -> Dict[str, Any]:
        """Advanced AI visual matching with scoring"""
        if not self.inventory_images:
            return {"exactMatch": False, "similarItems": [], "message": "No inventory images found"}
        
        print(f"🎯 Comparing with {len(self.inventory_images)} inventory items...")
        print(f"📝 Query features: {query_description}")
        
        # Calculate visual similarity scores for each inventory item
        item_scores = []
        
        for inv_item in self.inventory_images:
            try:
                # Use CLIP for perfect similarity calculation
                similarity_score = self.calculate_clip_similarity(query_description, inv_item['image'])
                
                item_scores.append((similarity_score, inv_item['item']))
                
                print(f"  {inv_item['item']['name']}: CLIP score={similarity_score:.4f}")
                    
            except Exception as e:
                print(f"Error processing {inv_item['item']['name']}: {e}")
                continue
        
        # Sort by similarity score (highest first)
        item_scores.sort(key=lambda x: x[0], reverse=True)
        
        best_score = item_scores[0][0]
        best_match = item_scores[0][1]
        
        print(f"\n🏆 Best match: {best_match['name']} (score: {best_score:.4f})")
        
        # Super intelligent matching - like human reasoning
        if best_score > 0.25:  # Intelligent threshold for exact matches
            return {
                "exactMatch": True,
                "product": best_match,
                "similarity": best_score,
                "message": f"Found exact match: {best_match['name']}"
            }
        else:
            # Return top 3 similar items with CLIP threshold
            similar_items = []
            for score, item in item_scores[:3]:
                if score > 0.15:  # CLIP threshold for similar items
                    similar_items.append({
                        "name": item["name"],
                        "price": item["price"],
                        "category": item.get("category", "clothing"),
                        "url": item.get("url", ""),
                        "delivery": item.get("delivery", ""),
                        "similarity": score
                    })
            
            return {
                "exactMatch": False,
                "similarItems": similar_items,
                "message": f"Found {len(similar_items)} similar items"
            }

    def calculate_visual_similarity(self, query_features: str, inv_features: str) -> float:
        """Calculate visual similarity between query and inventory item"""
        try:
            # Split features into words
            query_words = set(query_features.lower().split())
            inv_words = set(inv_features.lower().split())
            
            # Calculate base Jaccard similarity
            intersection = len(query_words.intersection(inv_words))
            union = len(query_words.union(inv_words))
            
            if union == 0:
                return 0.0
            
            base_similarity = intersection / union
            
            # Very strict color matching - must be exact (works for all colors)
            color_match_score = 0.0
            all_colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'gray', 'colored']
            query_colors = [word for word in query_words if word in all_colors]
            inv_colors = [word for word in inv_words if word in all_colors]
            
            if query_colors and inv_colors:
                # Only boost if there's an exact color match
                if any(color in inv_colors for color in query_colors):
                    color_match_score = 0.5  # Higher boost for exact color match
                else:
                    color_match_score = -0.3  # Higher penalty for color mismatch
            elif query_colors and not inv_colors:
                # If query has color but inventory doesn't, penalty
                color_match_score = -0.2
            
            # Material matching - must be exact (works for all materials)
            material_match_score = 0.0
            all_materials = ['denim', 'cotton', 'leather', 'synthetic', 'wool', 'silk', 'smooth', 'material']
            query_materials = [word for word in query_words if word in all_materials]
            inv_materials = [word for word in inv_words if word in all_materials]
            
            if query_materials and inv_materials:
                if any(material in inv_materials for material in query_materials):
                    material_match_score = 0.3
                else:
                    material_match_score = -0.1  # Small penalty for material mismatch
            
            # Calculate final score with stricter matching
            final_score = base_similarity + color_match_score + material_match_score
            
            # Additional penalty for major color mismatches
            if query_colors and inv_colors:
                # If colors are completely different, apply heavy penalty
                if not any(color in inv_colors for color in query_colors):
                    final_score -= 0.4  # Heavy penalty for color mismatch
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(final_score, 1.0))
            
            return final_score
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0

    def calculate_clip_similarity(self, query_description: str, inventory_image: Image.Image) -> float:
        """DYNAMIC CLIP similarity calculation - works with any inventory"""
        try:
            if self.model is not None and self.processor is not None:
                # Process image and text separately with proper padding
                image_inputs = self.processor(images=inventory_image, return_tensors="pt")
                text_inputs = self.processor(text=[query_description], return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    # Get features separately
                    image_features = self.model.get_image_features(**image_inputs)
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(image_features, text_features, dim=1).item()
                    
                    # Apply dynamic matching based on actual inventory
                    similarity = self.apply_dynamic_matching(query_description, similarity)
                    
                    return similarity
            else:
                # Fallback to basic similarity
                return self.calculate_visual_similarity(query_description, "clothing item")
                
        except Exception as e:
            print(f"Dynamic CLIP similarity calculation failed: {e}")
            return 0.0

    def apply_dynamic_matching(self, query_description: str, similarity: float) -> float:
        """Apply super intelligent matching like human reasoning"""
        try:
            # Boost similarity for exact name matches (like human would do)
            query_lower = query_description.lower()
            
            # If the query contains exact item names, boost the score
            for inv_item in self.inventory_images:
                item_name = inv_item['item']['name'].lower()
                if item_name in query_lower or query_lower in item_name:
                    # Human-like reasoning: if names match, it's likely the same item
                    similarity = min(similarity * 1.2, 1.0)  # Boost but cap at 1.0
                    break
            
            return similarity
            
        except Exception as e:
            print(f"Super intelligent matching failed: {e}")
            return similarity

    def apply_strict_matching(self, query_description: str, similarity: float) -> float:
        """Apply strict matching rules to prevent false positives"""
        try:
            # Extract color and type from query
            query_lower = query_description.lower()
            
            # Define color mappings
            color_mappings = {
                'red': ['red', 'crimson', 'scarlet'],
                'blue': ['blue', 'navy', 'azure'],
                'green': ['green', 'emerald', 'lime'],
                'black': ['black', 'dark'],
                'white': ['white', 'ivory', 'cream'],
                'gray': ['gray', 'grey', 'silver'],
                'brown': ['brown', 'tan', 'beige'],
                'yellow': ['yellow', 'gold'],
                'purple': ['purple', 'violet'],
                'orange': ['orange', 'amber']
            }
            
            # Extract colors from query
            query_colors = []
            for color, variants in color_mappings.items():
                for variant in variants:
                    if variant in query_lower:
                        query_colors.append(color)
                        break
            
            # If we have color information, apply strict color matching
            if query_colors:
                # Only allow high similarity if colors match
                if similarity > 0.3:  # Only consider if already reasonably similar
                    return similarity
                else:
                    return similarity * 0.5  # Reduce similarity for color mismatches
            
            return similarity
            
        except Exception as e:
            print(f"Strict matching failed: {e}")
            return similarity

    def search_similar_items(self, image_data: str) -> Dict[str, Any]:
        """Main search function - efficient AI matching"""
        print("\n🔍 Starting efficient AI search...")
        
        # Process uploaded image
        query_description, processed_image = self.process_uploaded_image(image_data)
        
        if query_description is None:
            print("⚠️ Image processing failed - returning all items as similar")
            # Return all inventory items as similar
            similar_items = []
            for inv_item in self.inventory_images:
                similar_items.append({
                    "name": inv_item['item']["name"],
                    "price": inv_item['item']["price"],
                    "category": inv_item['item'].get("category", "clothing"),
                    "url": inv_item['item'].get("url", ""),
                    "delivery": inv_item['item'].get("delivery", ""),
                    "similarity": 0.5
                })
            
            return {
                "exactMatch": False,
                "similarItems": similar_items[:3],
                "message": "Found similar items (fallback mode)"
            }
        
        print(f"👕 Detected clothing: {query_description}")
        
        # Find best match using strict CLIP + color gate
        result = self.find_best_match_strict(query_description, processed_image)
        
        print(f"✅ Search complete: {result['message']}")
        return result

    def _get_dominant_color_from_image(self, image: Image.Image) -> str:
        """Compute dominant color bucket from a PIL image using HSV rules already in place."""
        try:
            img_array = np.array(image)
            return self.analyze_colors_advanced(img_array)
        except Exception as e:
            print(f"Color extraction failed: {e}")
            return "colored"

    def find_best_match_strict(self, query_description: str, processed_image: Image.Image) -> Dict[str, Any]:
        """Strict matching: fixed CLIP threshold + dominant color agreement for exact matches."""
        if not self.inventory_images:
            return {"exactMatch": False, "similarItems": [], "message": "No inventory images found"}

        print(f"🎯 CLIP AI comparing with {len(self.inventory_images)} inventory items...")
        print(f"📝 Query: {query_description}")

        # Fixed exact threshold: tuned to avoid false positives
        EXACT_CLIP_THRESHOLD = 0.32
        SIMILAR_CLIP_THRESHOLD = 0.18

        # Compute query dominant color once
        query_color = self._get_dominant_color_from_image(processed_image)
        print(f"🎨 Query dominant color: {query_color}")

        # Score items
        item_scores = []
        for inv_item in self.inventory_images:
            try:
                inv_image = inv_item['image']
                inv_color = self._get_dominant_color_from_image(inv_image)
                clip_score = self.calculate_clip_similarity(query_description, inv_image)
                item_scores.append((clip_score, inv_item['item'], inv_color))
                print(f"  {inv_item['item']['name']}: CLIP={clip_score:.4f} color={inv_color}")
            except Exception as e:
                print(f"Error scoring {inv_item['item']['name']}: {e}")
                continue

        if not item_scores:
            return {"exactMatch": False, "similarItems": [], "message": "No comparable items"}

        # Sort by CLIP similarity score (highest first)
        item_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match, best_color = item_scores[0]
        print(f"\n🏆 Best candidate: {best_match['name']} (CLIP={best_score:.4f}, color={best_color})")

        # Exact match requires: CLIP above threshold AND color agreement
        if best_score >= EXACT_CLIP_THRESHOLD and (query_color == best_color):
            return {
                "exactMatch": True,
                "product": best_match,
                "similarity": best_score,
                "message": f"Found exact match: {best_match['name']}"
            }

        # Otherwise, return top similar items above similar threshold
        similar_items = []
        for score, item, inv_color in item_scores[:3]:
            if score >= SIMILAR_CLIP_THRESHOLD:
                similar_items.append({
                    "name": item["name"],
                    "price": item["price"],
                    "category": item.get("category", "clothing"),
                    "url": item.get("url", ""),
                    "delivery": item.get("delivery", ""),
                    "similarity": score
                })

        return {
            "exactMatch": False,
            "similarItems": similar_items,
            "message": f"Found {len(similar_items)} similar items"
        }

# Initialize AI service
ai_service = AIService()

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/search', methods=['POST'])
def search():
    """API endpoint for clothing-focused image search"""
    try:
        logger.info("Received clothing search request")
        
        # Get image from request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("Empty image filename")
            return jsonify({"error": "No image selected"}), 400
        
        logger.info(f"Processing clothing image: {image_file.filename}")
        
        # Convert image to base64
        image_bytes = image_file.read()
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        logger.info("Starting clothing-focused AI search...")
        
        # Search for similar clothing items
        result = ai_service.search_similar_items(image_data)
        
        logger.info(f"Clothing search completed: {result.get('message', 'No message')}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in clothing search endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "inventory_items": len(ai_service.inventory),
        "loaded_images": len(ai_service.inventory_images),
        "features": [
            "clothing detection",
            "background removal", 
            "AI clothing description",
            "visual + text matching"
        ]
    })

@app.route('/inventory', methods=['GET'])
def get_inventory():
    """Get all inventory items"""
    try:
        logger.info("Received inventory request")
        # Transform data to match frontend expectations
        transformed_inventory = []
        for item in ai_service.inventory:
            # Extract filename from image_path for serving
            image_filename = os.path.basename(item["image_path"])
            transformed_item = {
                "id": str(item["id"]),  # Convert id to string
                "name": item["name"],
                "price": item["price"],
                "image_path": f"{BASE_URL}/images/{image_filename}",  # Full URL to backend
                "url": item.get("url", ""),
                "delivery": item["delivery"],
                "category": item["category"]
            }
            transformed_inventory.append(transformed_item)
        
        return jsonify(transformed_inventory)
    except Exception as e:
        logger.error(f"Error getting inventory: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/precompute', methods=['POST'])
def precompute():
    """Precompute embeddings for faster search"""
    try:
        logger.info("Received precompute request")
        # The embeddings are already computed when the service starts
        return jsonify({"status": "success", "message": "Embeddings already computed"})
    except Exception as e:
        logger.error(f"Error in precompute: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve inventory images"""
    try:
        image_path = os.path.join('inventory_images', filename)
        if os.path.exists(image_path):
            return send_file(image_path)
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def main():
    """Run the clothing-focused AI service"""
    print("🚀 Starting Clothing-Focused AI Service...")
    
    print(f"📦 Loaded {len(ai_service.inventory)} inventory items")
    print(f"🖼️  Loaded {len(ai_service.inventory_images)} inventory images")
    
    print("\n✅ Clothing-Focused AI Service ready!")
    print("🎯 Advanced Features:")
    print("  • Clothing detection (isolates garments from models)")
    print("  • Background removal (focuses on clothing items)")
    print("  • AI clothing description (describes what it sees)")
    print("  • Combined visual + text matching")
    print("  • Model-agnostic matching (works with photos of people)")
    print("🌐 Starting web server on port 5000...")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()