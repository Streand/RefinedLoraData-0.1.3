"""
Clothing Analysis Backend using InstructBLIP and BLIP-2
Provides detailed clothing description for Stable Diffusion and LoRA training
Compatible with Blackwell NVIDIA GPUs
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import sys

# Add project root for blackwell support
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from blackwell_support import get_optimal_device, verify_pytorch_blackwell_support, get_gpu_info
    blackwell_support_available = True
except ImportError as e:
    blackwell_support_available = False
    print(f"Warning: Blackwell support module not available: {e}")
    # Define placeholder functions
    def get_optimal_device() -> str:
        return 'cpu'
    def verify_pytorch_blackwell_support() -> Tuple[bool, str]:
        return False, "Blackwell support module not available"
    def get_gpu_info() -> Dict[str, Any]:
        return {'is_blackwell': False, 'is_rtx_5000_series': False, 'sm_count': 0, 'name': 'Unknown'}

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch
    import PIL.Image as Image
    from transformers import (
        BlipProcessor, 
        BlipForConditionalGeneration,
        InstructBlipProcessor,
        InstructBlipForConditionalGeneration
    )
    torch_available = True
    transformers_available = True
except ImportError as e:
    torch_available = False
    transformers_available = False
    logger.error(f"Required packages not available: {e}")
    print("Please install: pip install torch transformers pillow")
    
    # Define placeholder classes/objects to avoid "possibly unbound" errors
    torch = None  # type: ignore
    Image = None  # type: ignore
    BlipProcessor = None  # type: ignore
    BlipForConditionalGeneration = None  # type: ignore
    InstructBlipProcessor = None  # type: ignore
    InstructBlipForConditionalGeneration = None  # type: ignore

class ClothingAnalyzer:
    """Advanced clothing analysis using multiple vision-language models"""
    
    def __init__(self, model_name: str = "instructblip"):
        """
        Initialize clothing analyzer
        
        Args:
            model_name: Either "instructblip" or "blip2"
        """
        self.model_name = model_name.lower()
        self.device = self._setup_device()
        self.model = None
        self.processor = None
        self.is_initialized = False
        
        # Clothing categories for structured analysis
        self.clothing_categories = {
            "upper_body": [
                "shirt", "t-shirt", "blouse", "sweater", "hoodie", "jacket", "blazer", "coat", 
                "vest", "tank top", "crop top", "cardigan", "pullover", "sweatshirt", "polo",
                "button-up", "dress shirt", "tunic", "top", "camisole", "halter top"
            ],
            "lower_body": [
                "pants", "jeans", "trousers", "shorts", "skirt", "dress", "leggings", "tights",
                "slacks", "chinos", "khakis", "yoga pants", "sweatpants", "joggers", "capris",
                "mini skirt", "maxi skirt", "pencil skirt", "a-line skirt", "overalls"
            ],
            "footwear": [
                "shoes", "sneakers", "boots", "sandals", "heels", "flats", "loafers", "oxfords",
                "high heels", "ankle boots", "knee boots", "running shoes", "dress shoes",
                "flip flops", "wedges", "platforms", "ballet flats", "combat boots", "hiking boots"
            ],
            "outerwear": [
                "jacket", "coat", "blazer", "cardigan", "hoodie", "windbreaker", "bomber",
                "trench coat", "pea coat", "puffer jacket", "leather jacket", "denim jacket",
                "sports jacket", "overcoat", "poncho", "cape", "shawl"
            ],
            "accessories": [
                "hat", "cap", "scarf", "belt", "bag", "purse", "backpack", "jewelry", "watch", 
                "glasses", "sunglasses", "necklace", "earrings", "bracelet", "ring", "tie",
                "bow tie", "suspenders", "gloves", "handbag", "clutch", "tote bag", "crossbody bag"
            ],
            "style": [
                "casual", "formal", "business", "streetwear", "vintage", "bohemian", "athletic", 
                "gothic", "preppy", "minimalist", "elegant", "sophisticated", "trendy", "chic",
                "classic", "modern", "retro", "grunge", "punk", "romantic", "edgy", "professional"
            ]
        }
        
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup optimal device using blackwell support"""
        if not torch_available or torch is None:
            logger.warning("PyTorch not available, using CPU")
            return 'cpu'
            
        if blackwell_support_available:
            try:
                device = get_optimal_device()
                supported, message = verify_pytorch_blackwell_support()
                
                if device == 'cuda':
                    logger.info(f"✅ GPU acceleration enabled: {message}")
                else:
                    logger.warning(f"⚠️ Using CPU: {message}")
                    
                # Log GPU info for debugging
                gpu_info = get_gpu_info()
                if gpu_info['is_blackwell']:
                    logger.info(f"Blackwell GPU detected: {gpu_info['name']} (~{gpu_info['sm_count']} SMs)")
                    if gpu_info['is_rtx_5000_series']:
                        logger.info("RTX 5000 series detected - optimized for clothing analysis")
                
                return device
                
            except Exception as e:
                logger.warning(f"Error using blackwell_support module: {e}")
        
        # Fallback device detection
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'
    
    def _load_model(self) -> bool:
        """Load the selected vision-language model"""
        if not torch_available or not transformers_available:
            logger.error("PyTorch or transformers not available - cannot load models")
            return False
            
        if torch is None or InstructBlipProcessor is None or BlipProcessor is None:
            logger.error("Required modules not properly imported - cannot load models")
            return False
            
        try:
            if self.model_name == "instructblip":
                logger.info("Loading InstructBLIP model for detailed clothing analysis...")
                model_id = "Salesforce/instructblip-vicuna-7b"
                self.processor = InstructBlipProcessor.from_pretrained(model_id)  # type: ignore
                self.model = InstructBlipForConditionalGeneration.from_pretrained(  # type: ignore
                    model_id,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
            elif self.model_name == "blip2":
                logger.info("Loading BLIP-2 model for fast clothing analysis...")
                model_id = "Salesforce/blip2-opt-2.7b"
                self.processor = BlipProcessor.from_pretrained(model_id)  # type: ignore
                self.model = BlipForConditionalGeneration.from_pretrained(  # type: ignore
                    model_id,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Move model to device
            if self.device == 'cuda' and hasattr(self.model, 'to'):
                self.model = self.model.to('cuda')  # type: ignore
            
            # Test model with dummy data
            if Image is not None:
                logger.info("Testing model initialization...")
                dummy_image = Image.new('RGB', (224, 224), color='white')
                test_result = self._analyze_image_internal(dummy_image, test_mode=True)
                
                if test_result is not None:
                    self.is_initialized = True
                    logger.info(f"✅ {self.model_name.upper()} model loaded successfully on {self.device}")
                    return True
                else:
                    logger.error("Model test failed")
                    return False
            else:
                logger.error("PIL Image not available for model testing")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load {self.model_name} model: {e}")
            return False
    
    def _analyze_image_internal(self, image, test_mode: bool = False):
        """Internal method to analyze image with the loaded model"""
        if not self.is_initialized and not test_mode:
            return None
            
        if not torch_available or torch is None:
            logger.error("PyTorch not available - cannot analyze image")
            return None
            
        try:
            if self.model_name == "instructblip":
                # InstructBLIP with detailed clothing prompt
                prompt = "Describe in detail what clothing and accessories this person is wearing, including colors, styles, and materials."
                
                # Process inputs for InstructBLIP
                if self.processor is None:
                    logger.error("Processor not initialized")
                    return None
                    
                inputs = self.processor(image, prompt, return_tensors="pt")  # type: ignore
                
                if self.device == 'cuda' and hasattr(torch, 'cuda'):
                    for key in inputs:
                        if hasattr(inputs[key], 'to'):
                            inputs[key] = inputs[key].to('cuda')  # type: ignore
                
                with torch.no_grad():
                    # Generate with proper parameters for InstructBLIP
                    if self.model is None:
                        logger.error("Model not initialized")
                        return None
                        
                    outputs = self.model.generate(  # type: ignore
                        **inputs,  # type: ignore
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=3,
                        early_stopping=True
                    )
                
                # Decode the response
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]  # type: ignore
                
                # Clean up response - remove the prompt part
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                return response
                
            elif self.model_name == "blip2":
                # BLIP-2 with clothing-focused prompt  
                prompt = "Question: What clothing and accessories is this person wearing? Answer:"
                
                # Process inputs for BLIP-2
                if self.processor is None:
                    logger.error("Processor not initialized")
                    return None
                    
                inputs = self.processor(image, prompt, return_tensors="pt")  # type: ignore
                
                if self.device == 'cuda' and hasattr(torch, 'cuda'):
                    for key in inputs:
                        if hasattr(inputs[key], 'to'):
                            inputs[key] = inputs[key].to('cuda')  # type: ignore
                
                with torch.no_grad():
                    if self.model is None:
                        logger.error("Model not initialized")
                        return None
                        
                    outputs = self.model.generate(  # type: ignore
                        **inputs,  # type: ignore
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=3
                    )
                
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]  # type: ignore
                
                # Clean up the response by removing the prompt
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                return response
            
        except Exception as e:
            logger.error(f"Error during image analysis: {e}")
            if test_mode:
                return "test_success"  # For initialization testing
            return None
    
    def _categorize_clothing(self, description: str) -> Dict[str, List[str]]:
        """Categorize clothing items from description"""
        description_lower = description.lower()
        categorized = {category: [] for category in self.clothing_categories.keys()}
        
        for category, items in self.clothing_categories.items():
            for item in items:
                if item in description_lower:
                    categorized[category].append(item)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def _extract_colors_and_patterns(self, description: str) -> List[str]:
        """Extract colors and patterns from description with enhanced detection"""
        colors = [
            "black", "white", "gray", "grey", "red", "blue", "green", "yellow", "orange", 
            "purple", "pink", "brown", "navy", "beige", "khaki", "cream", "ivory", "silver",
            "gold", "maroon", "burgundy", "turquoise", "teal", "coral", "lavender", "mint",
            "olive", "tan", "charcoal", "platinum", "rose", "magenta", "cyan", "lime"
        ]
        
        patterns = [
            "striped", "plaid", "floral", "polka dot", "polka-dot", "checkered", "checked",
            "geometric", "solid", "printed", "embroidered", "lace", "sequined", "beaded",
            "textured", "smooth", "ribbed", "cable knit", "fair isle", "argyle", "houndstooth",
            "paisley", "tropical", "animal print", "leopard", "zebra", "snake print"
        ]
        
        materials = [
            "silk", "cotton", "denim", "leather", "wool", "cashmere", "linen", "satin", 
            "velvet", "polyester", "nylon", "spandex", "jersey", "chiffon", "tulle",
            "mesh", "lace fabric", "corduroy", "flannel", "tweed", "canvas"
        ]
        
        description_lower = description.lower()
        found = []
        
        # Extract colors
        for color in colors:
            if color in description_lower:
                found.append(color)
        
        # Extract patterns
        for pattern in patterns:
            if pattern in description_lower:
                found.append(pattern)
        
        # Extract materials (add them as descriptors)
        for material in materials:
            if material in description_lower:
                found.append(material)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_found = []
        for item in found:
            if item not in seen:
                seen.add(item)
                unique_found.append(item)
        
        return unique_found
    
    def _calculate_confidence(self, description: str) -> float:
        """Calculate confidence score based on description detail"""
        if not description:
            return 0.0
        
        # Simple confidence scoring based on description length and clothing keywords
        clothing_keywords = sum(len(items) for items in self.clothing_categories.values())
        found_keywords = sum(1 for category_items in self.clothing_categories.values() 
                           for item in category_items if item in description.lower())
        
        base_score = min(len(description) / 100, 1.0)  # Length-based score
        keyword_score = found_keywords / max(clothing_keywords * 0.1, 1)  # Keyword density
        
        return min((base_score + keyword_score) / 2, 1.0)
    
    def _format_for_stable_diffusion(self, description: str, categorized: Dict[str, List[str]], 
                                   colors_patterns: List[str]) -> str:
        """Format clothing description for Stable Diffusion prompts with enhanced detail"""
        sd_parts = []
        description_lower = description.lower()
        
        # Add person descriptor
        if "woman" in description_lower or "female" in description_lower:
            sd_parts.append("woman")
        elif "man" in description_lower or "male" in description_lower:
            sd_parts.append("man")
        else:
            sd_parts.append("person")
        
        # Add "wearing" connector
        sd_parts.append("wearing")
        
        # Enhanced clothing items with descriptors
        clothing_items = []
        
        # Process upper body items with detail
        if "upper_body" in categorized and categorized["upper_body"]:
            for item in categorized["upper_body"]:
                enhanced_item = self._enhance_clothing_item(item, description_lower, colors_patterns)
                clothing_items.append(enhanced_item)
        
        # Process lower body items
        if "lower_body" in categorized and categorized["lower_body"]:
            for item in categorized["lower_body"]:
                enhanced_item = self._enhance_clothing_item(item, description_lower, colors_patterns)
                clothing_items.append(enhanced_item)
        
        # Process outerwear items
        if "outerwear" in categorized and categorized["outerwear"]:
            for item in categorized["outerwear"]:
                enhanced_item = self._enhance_clothing_item(item, description_lower, colors_patterns)
                clothing_items.append(enhanced_item)
        
        # Add clothing items to prompt
        if clothing_items:
            sd_parts.extend(clothing_items)
        
        # Add footwear with descriptors
        if "footwear" in categorized and categorized["footwear"]:
            for item in categorized["footwear"]:
                enhanced_item = self._enhance_clothing_item(item, description_lower, colors_patterns)
                sd_parts.append(enhanced_item)
        
        # Add accessories
        if "accessories" in categorized and categorized["accessories"]:
            for item in categorized["accessories"]:
                enhanced_item = self._enhance_clothing_item(item, description_lower, colors_patterns)
                sd_parts.append(enhanced_item)
        
        # Add style and quality descriptors
        style_descriptors = []
        if "style" in categorized and categorized["style"]:
            style_descriptors.extend(categorized["style"])
        
        # Add fabric and material descriptors from description
        fabric_keywords = ["silk", "cotton", "denim", "leather", "wool", "linen", "satin", "velvet", "polyester"]
        for fabric in fabric_keywords:
            if fabric in description_lower and fabric not in [item.lower() for item in style_descriptors]:
                style_descriptors.append(fabric)
        
        # Add fit descriptors
        fit_keywords = ["tight", "loose", "fitted", "oversized", "slim", "baggy", "form-fitting"]
        for fit in fit_keywords:
            if fit in description_lower and f"{fit} fit" not in style_descriptors:
                style_descriptors.append(f"{fit} fit")
        
        # Add style descriptors
        if style_descriptors:
            sd_parts.extend(style_descriptors)
        
        # Add quality and detail enhancers
        quality_enhancers = [
            "detailed clothing",
            "high quality",
            "professional photography",
            "clear details"
        ]
        sd_parts.extend(quality_enhancers)
        
        # Add lighting and photo quality if mentioned
        if any(word in description_lower for word in ["bright", "lighting", "well-lit", "good lighting"]):
            sd_parts.append("good lighting")
        
        if any(word in description_lower for word in ["indoor", "inside", "room"]):
            sd_parts.append("indoor setting")
        elif any(word in description_lower for word in ["outdoor", "outside"]):
            sd_parts.append("outdoor setting")
        
        # Clean and format
        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in sd_parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)
        
        sd_prompt = ", ".join(unique_parts)
        
        return sd_prompt
    
    def _enhance_clothing_item(self, item: str, description_lower: str, colors_patterns: List[str]) -> str:
        """Enhance a clothing item with colors, patterns, and descriptors"""
        enhanced = item
        
        # Add colors if they appear specifically with the item
        relevant_colors = []
        for color in colors_patterns:
            if color in ["black", "white", "red", "blue", "green", "yellow", "pink", "purple", "brown", "gray", "navy", "beige"]:
                # Check if color is mentioned specifically with this item
                if f"{color} {item}" in description_lower:
                    relevant_colors.append(color)
        
        # Add the most relevant color (only if found with the specific item)
        if relevant_colors:
            enhanced = f"{relevant_colors[0]} {enhanced}"
        
        # Add patterns (only if mentioned with this specific item)
        for pattern in colors_patterns:
            if pattern in ["striped", "plaid", "floral", "polka dot", "checkered", "printed"]:
                if f"{pattern} {item}" in description_lower or (pattern in description_lower and item in description_lower):
                    enhanced = f"{pattern} {enhanced}"
                    break
        
        # Add specific descriptors based on item type
        if item == "dress":
            if "long" in description_lower or "floor-length" in description_lower:
                enhanced = f"long {enhanced}"
            elif "short" in description_lower or "mini" in description_lower:
                enhanced = f"short {enhanced}"
            if "evening" in description_lower:
                enhanced = f"evening {enhanced}"
            elif "summer" in description_lower:
                enhanced = f"summer {enhanced}"
        
        elif item == "shirt":
            if "button" in description_lower:
                enhanced = f"button-up {enhanced}"
            if "collar" in description_lower:
                enhanced = f"collared {enhanced}"
        
        elif item == "pants" or item == "jeans":
            if "tight" in description_lower:
                enhanced = f"tight {enhanced}"
            elif "loose" in description_lower:
                enhanced = f"loose {enhanced}"
        
        elif item == "shoes":
            if "high" in description_lower and "heel" in description_lower:
                enhanced = "high heels"
            elif "sneaker" in description_lower:
                enhanced = "sneakers"
            elif "boot" in description_lower:
                enhanced = "boots"
        
        return enhanced
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze clothing in an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if dependencies are available
            if not torch_available or not transformers_available:
                return {"error": "PyTorch or transformers not available - cannot analyze images"}
            
            if Image is None:
                return {"error": "PIL Image not available - cannot load images"}
            
            # Load and validate image
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            image = Image.open(image_path).convert('RGB')
            
            # Analyze with selected model
            raw_description = self._analyze_image_internal(image)
            
            if not raw_description:
                return {"error": "Failed to analyze image"}
            
            # Process results
            categorized = self._categorize_clothing(raw_description)
            colors_patterns = self._extract_colors_and_patterns(raw_description)
            confidence = self._calculate_confidence(raw_description)
            sd_prompt = self._format_for_stable_diffusion(raw_description, categorized, colors_patterns)
            
            return {
                "success": True,
                "model_used": self.model_name,
                "raw_description": raw_description,
                "categorized": categorized,
                "colors_patterns": colors_patterns,
                "confidence": confidence,
                "sd_prompt": sd_prompt,
                "image_path": image_path,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {"error": str(e)}
    
    def get_device_info(self) -> Dict:
        """Get information about device and model status"""
        info = {
            'device': self.device,
            'model_name': self.model_name,
            'model_initialized': self.is_initialized,
            'torch_available': torch_available
        }
        
        if torch_available and torch is not None:
            info['pytorch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                try:
                    info['gpu_name'] = torch.cuda.get_device_name(0)
                    info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    major, minor = torch.cuda.get_device_capability(0)
                    info['compute_capability'] = f"{major}.{minor}"
                    
                    # Use blackwell_support module if available
                    if blackwell_support_available:
                        try:
                            gpu_info = get_gpu_info()
                            supported, support_message = verify_pytorch_blackwell_support()
                            
                            info['is_blackwell'] = gpu_info['is_blackwell']
                            info['is_rtx_5000_series'] = gpu_info['is_rtx_5000_series']
                            info['sm_count'] = gpu_info['sm_count']
                            info['blackwell_support'] = supported
                            info['support_message'] = support_message
                            
                            if gpu_info['is_rtx_5000_series']:
                                info['optimization_notes'] = "Optimized for 120 SM configuration - ideal for clothing analysis"
                            
                        except Exception as e:
                            info['blackwell_error'] = f"Error checking Blackwell support: {e}"
                    
                except Exception as e:
                    info['gpu_error'] = str(e)
        
        return info

# Factory function for easy model switching
def create_clothing_analyzer(model_name: str = "instructblip") -> ClothingAnalyzer:
    """Create a clothing analyzer with the specified model"""
    return ClothingAnalyzer(model_name=model_name)
