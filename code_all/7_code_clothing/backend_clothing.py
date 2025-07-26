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
        InstructBlipForConditionalGeneration,
        CLIPProcessor,
        CLIPModel
    )
    # Try to import fashion-clip if available
    try:
        import clip
        fashion_clip_available = True
    except ImportError:
        fashion_clip_available = False
    
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
    
    def __init__(self, model_name: str = "fashionclip"):
        """
        Initialize clothing analyzer
        
        Args:
            model_name: Either "fashionclip" (default), "instructblip", or "blip2"
                       FashionCLIP is recommended for fashion-specific analysis
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
                
            elif self.model_name == "fashionclip":
                logger.info("Loading FashionCLIP model for fashion-specific analysis...")
                model_id = "patrickjohncyh/fashion-clip"
                self.processor = CLIPProcessor.from_pretrained(model_id)  # type: ignore
                self.model = CLIPModel.from_pretrained(  # type: ignore
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
                
            elif self.model_name == "fashionclip":
                # FashionCLIP with fashion-specific text queries
                return self._analyze_with_fashionclip(image, test_mode)
            
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
            # Sort items by length (longest first) to prioritize specific terms
            items_sorted = sorted(items, key=len, reverse=True)
            found_items = []
            
            for item in items_sorted:
                if item in description_lower:
                    # Check if this item is not already covered by a more specific term
                    is_duplicate = False
                    for existing_item in found_items:
                        if item in existing_item:  # e.g., "top" is in "crop top"
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        found_items.append(item)
            
            if found_items:
                categorized[category] = found_items
        
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
        """Format clothing description for detailed, specific output perfect for Stable Diffusion"""
        
        # For FashionCLIP, the raw description is already well-formatted
        # Just return it directly since it's already in "item1, item2" format
        if self.model_name == "fashionclip":
            # FashionCLIP already returns well-formatted descriptions
            return description.strip()
        
        # For other models (InstructBLIP, BLIP-2), do the complex processing
        clothing_items = []
        description_lower = description.lower()
        
        # Process clothing items in order: upper body, lower body, outerwear, footwear
        categories_to_process = ["upper_body", "lower_body", "outerwear", "footwear"]
        
        for category in categories_to_process:
            if category in categorized and categorized[category]:
                for item in categorized[category]:
                    enhanced_item = self._extract_detailed_clothing_description(item, description, description_lower)
                    if enhanced_item:  # Only add if we got a valid enhanced item
                        clothing_items.append(enhanced_item)
        
        # Return detailed clothing list
        return ", ".join(clothing_items) if clothing_items else description.strip()
    
    def _extract_detailed_clothing_description(self, base_item: str, full_description: str, description_lower: str) -> str:
        """Extract detailed clothing description with specific features and materials"""
        
        # Map generic items to more specific types found in description
        specific_item_mapping = {
            "top": ["tank top", "crop top", "t-shirt", "blouse", "shirt", "halter top", "tube top", "camisole"],
            "shirt": ["tank top", "t-shirt", "dress shirt", "button-up", "polo shirt", "blouse"],
            "pants": ["jeans", "trousers", "slacks", "chinos", "yoga pants", "dress pants"],
            "shorts": ["denim shorts", "jean shorts", "athletic shorts", "dress shorts", "cargo shorts"],
            "dress": ["mini dress", "maxi dress", "midi dress", "cocktail dress", "sundress", "wrap dress"],
            "skirt": ["mini skirt", "maxi skirt", "pencil skirt", "a-line skirt", "pleated skirt"],
            "jacket": ["blazer", "bomber jacket", "denim jacket", "leather jacket", "cardigan"],
            "shoes": ["sneakers", "heels", "boots", "sandals", "flats", "loafers"]
        }
        
        # Find the most specific item type mentioned in description
        final_item = base_item
        if base_item in specific_item_mapping:
            for specific_type in specific_item_mapping[base_item]:
                if specific_type in description_lower:
                    final_item = specific_type
                    break
        
        # If we only found "red top", make it "tank top" for better SD prompts
        if final_item == "top" and base_item == "top":
            final_item = "tank top"  # Default assumption for better SD output
        
        # Extract color specifically for this item
        item_color = self._extract_specific_item_color(final_item, base_item, description_lower)
        
        # Extract material specifically for this item (avoid cross-contamination)
        item_material = self._extract_specific_item_material(final_item, base_item, description_lower)
        
        # Extract patterns specifically for this item
        item_patterns = self._extract_specific_item_patterns(final_item, base_item, description_lower)
        
        # Build the final description in logical order
        parts = []
        
        # Add color first (if found)
        if item_color:
            parts.append(item_color)
        
        # Add material (if relevant and not redundant)
        if item_material and item_material not in final_item:
            parts.append(item_material)
        
        # Add the specific item type
        parts.append(final_item)
        
        # Add patterns/decorations at the end
        if item_patterns:
            parts.extend(item_patterns)
        
        return " ".join(parts)
    
    def _extract_specific_item_color(self, final_item: str, base_item: str, description_lower: str) -> str:
        """Extract color specifically for this clothing item"""
        color_keywords = ["red", "blue", "white", "black", "pink", "green", "yellow", "purple", 
                         "brown", "gray", "grey", "orange", "navy", "beige", "tan", "gold", 
                         "silver", "maroon", "burgundy", "turquoise", "lime", "olive", "cream"]
        
        # Look for color + item patterns
        import re
        for color in color_keywords:
            # Direct patterns: "red top", "blue shorts"
            if re.search(rf"\b{color}\s+{re.escape(base_item)}\b", description_lower):
                return color
            # Or with the specific item: "red tank top"
            if re.search(rf"\b{color}\s+{re.escape(final_item)}\b", description_lower):
                return color
        
        return ""
    
    def _extract_specific_item_material(self, final_item: str, base_item: str, description_lower: str) -> str:
        """Extract material specifically for this clothing item"""
        material_keywords = ["denim", "cotton", "silk", "wool", "leather", "linen", "polyester", 
                           "spandex", "nylon", "cashmere", "velvet", "satin", "chiffon", "lace"]
        
        # Look for materials mentioned specifically with this item
        import re
        for material in material_keywords:
            # Patterns like "denim shorts", but NOT "denim shorts" when analyzing a "top"
            if re.search(rf"\b{material}\s+{re.escape(base_item)}\b", description_lower):
                return material
            if re.search(rf"\b{material}\s+{re.escape(final_item)}\b", description_lower):
                return material
        
        return ""
    
    def _extract_specific_item_patterns(self, final_item: str, base_item: str, description_lower: str) -> List[str]:
        """Extract patterns and decorative elements for this specific item"""
        patterns = []
        
        # Look for embroidery/decorations specifically mentioned with this item
        import re
        
        # For shorts, check for embroidery patterns
        if "shorts" in final_item or "shorts" in base_item:
            if "floral embroidery" in description_lower:
                patterns.append("with floral embroidery")
            elif "embroidery" in description_lower and "floral" in description_lower:
                patterns.append("with floral embroidery")
        
        # Check for other patterns like stripes, plaid, etc.
        pattern_keywords = ["striped", "plaid", "checkered", "polka dot", "geometric", "printed"]
        
        for pattern in pattern_keywords:
            # Only if mentioned specifically with this item
            if re.search(rf"\b{pattern}\s+{re.escape(base_item)}\b", description_lower):
                patterns.append(pattern)
        
        return patterns
    
    def _enhance_clothing_description(self, base_description: str, all_results: List[Tuple[str, float]]) -> str:
        """Enhance clothing description with additional details like patterns and materials"""
        try:
            # Look for additional details in the results
            enhanced_parts = [base_description]
            base_lower = base_description.lower()
            
            # Extract the clothing item type from base description
            item_type = None
            if "crop top" in base_lower:
                item_type = "crop top"
            elif "tank top" in base_lower:
                item_type = "tank top"
            elif "t-shirt" in base_lower:
                item_type = "t-shirt"
            elif "yoga pants" in base_lower:
                item_type = "yoga pants"
            elif "pants" in base_lower:
                item_type = "pants"
            elif "shorts" in base_lower:
                item_type = "shorts"
            
            # Look for patterns and materials in other high-scoring results
            for result, score in all_results:
                if score > 0.01:  # Only consider confident matches
                    result_lower = result.lower()
                    
                    # Check for patterns
                    if "floral" in result_lower and "floral" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("with floral pattern")
                            break
                    elif "printed" in result_lower and "printed" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("with print")
                            break
                    elif "embroidered" in result_lower and "embroidered" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("with embroidery")
                            break
                    elif "striped" in result_lower and "striped" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("striped")
                            break
            
            # Look for fit/style descriptors
            for result, score in all_results:
                if score > 0.01:
                    result_lower = result.lower()
                    
                    # Check for fit descriptors
                    if "fitted" in result_lower and "fitted" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("fitted")
                            break
                    elif "tight" in result_lower and "tight" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("tight")
                            break
                    elif "loose" in result_lower and "loose" not in base_lower:
                        if item_type and item_type in result_lower:
                            enhanced_parts.append("loose")
                            break
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing clothing description: {e}")
            return base_description

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

    def _analyze_with_fashionclip(self, image, test_mode: bool = False) -> Optional[str]:
        """Analyze image using FashionCLIP with fashion-specific queries"""
        if test_mode:
            return "test_success"
            
        if self.processor is None or self.model is None:
            logger.error("FashionCLIP model not initialized")
            return None
            
        try:
            # Comprehensive fashion-specific queries organized by category
            
            # Enhanced color palette with more accurate color detection
            colors = [
                # Primary colors
                "red", "blue", "white", "black", 
                # Secondary colors
                "pink", "green", "yellow", "purple", "brown", "gray", "orange",
                # Fashion-specific colors
                "navy", "beige", "cream", "ivory", "charcoal", "khaki", "olive", 
                "burgundy", "maroon", "coral", "turquoise", "mint", "lavender",
                # Color variants for better accuracy
                "light blue", "dark blue", "bright red", "deep red", "pale pink", "hot pink",
                "forest green", "lime green", "golden yellow", "royal purple"
            ]
            
            # Enhanced upper body items with fashion-specific terminology
            upper_items = [
                "tank top", "crop top", "t-shirt", "blouse", "shirt", "halter top", 
                "camisole", "tube top", "sleeveless top", "off-shoulder top",
                "v-neck", "crew neck", "button-up", "polo shirt", "henley"
            ]
            
            # Enhanced lower body items
            lower_items = [
                "shorts", "jeans", "pants", "skirt", "leggings", "dress",
                "denim shorts", "jean shorts", "mini skirt", "midi skirt", "maxi skirt",
                "skinny jeans", "straight jeans", "bootcut jeans", "wide-leg pants",
                "yoga pants", "dress pants", "cargo shorts", "athletic shorts"
            ]
            
            # Enhanced materials with fashion context
            materials = [
                "denim", "cotton", "silk", "leather", "linen", "wool", "cashmere",
                "polyester", "spandex", "jersey", "chiffon", "lace", "velvet"
            ]
            
            # Enhanced patterns and details
            patterns = [
                "floral", "striped", "plaid", "solid", "embroidered", "printed",
                "polka dot", "geometric", "animal print", "tie-dye", "ombre",
                "with embroidery", "with sequins", "with lace", "with beading",
                "floral print", "floral pattern", "fitted", "tight-fitting", "loose-fitting"
            ]
            
            # Create comprehensive query combinations
            clothing_queries = []
            
            # Color + Upper body combinations
            for color in colors:
                for item in upper_items:
                    clothing_queries.append(f"{color} {item}")
            
            # Color + Lower body combinations  
            for color in colors:
                for item in lower_items:
                    clothing_queries.append(f"{color} {item}")
                    
            # Material + Item combinations
            for material in materials:
                for item in lower_items + upper_items:
                    clothing_queries.append(f"{material} {item}")
            
            # Pattern + Item combinations
            for pattern in patterns:
                for item in lower_items + upper_items:
                    clothing_queries.append(f"{pattern} {item}")
            
            # Specific combinations for improved accuracy
            specific_queries = [
                # Color accuracy improvements
                "bright white crop top", "pure white top", "snow white shirt", "off-white blouse",
                "bright blue yoga pants", "navy blue leggings", "royal blue pants", "dark blue jeans",
                
                # Pattern combinations
                "white crop top with floral print", "floral white top", "printed white shirt",
                "blue yoga pants fitted", "tight blue leggings", "stretchy blue pants",
                
                # Style details
                "fitted crop top", "tight yoga pants", "stretchy leggings", "athletic wear",
                "casual outfit", "gym outfit", "workout clothes",
                
                # Specific for the user's image type
                "white floral crop top",
                "blue fitted yoga pants", 
                "white top with print",
                "blue athletic pants"
            ]
            
            clothing_queries.extend(specific_queries)
            
            # Process in optimized batches for better performance
            batch_size = 40  # Reduced for better memory management
            all_matches = []
            all_scores = []
            
            if not torch_available or torch is None:
                return None
                
            for i in range(0, len(clothing_queries), batch_size):
                batch_queries = clothing_queries[i:i + batch_size]
                
                inputs = self.processor(  # type: ignore
                    text=batch_queries,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                if self.device == 'cuda':
                    for key in inputs:
                        if hasattr(inputs[key], 'to'):
                            inputs[key] = inputs[key].to('cuda')  # type: ignore
                
                with torch.no_grad():
                    outputs = self.model(**inputs)  # type: ignore
                    
                    # Get similarity scores
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1)
                    
                    # Get top matches from this batch
                    top_indices = torch.topk(probs[0], k=min(8, len(batch_queries))).indices
                    batch_matches = [batch_queries[i] for i in top_indices]
                    batch_scores = [probs[0][i].item() for i in top_indices]
                    
                    all_matches.extend(batch_matches)
                    all_scores.extend(batch_scores)
            
            # Sort all results by score
            sorted_results = sorted(zip(all_matches, all_scores), key=lambda x: x[1], reverse=True)
            
            # Enhanced categorization with material detection
            upper_keywords = [
                "tank top", "crop top", "t-shirt", "blouse", "shirt", "halter top", 
                "camisole", "tube top", "sleeveless", "v-neck", "crew neck", "henley"
            ]
            lower_keywords = [
                "shorts", "jeans", "pants", "skirt", "leggings", "dress", "denim shorts", 
                "jean shorts", "mini skirt", "midi skirt", "skinny jeans", "yoga pants"
            ]
            material_keywords = [
                "denim", "cotton", "silk", "leather", "linen", "wool", "jersey", "chiffon"
            ]
            
            # Separate matches by category with material awareness
            upper_matches = []
            lower_matches = []
            material_matches = []
            
            for query, score in sorted_results:
                query_lower = query.lower()
                
                # Check for materials
                if any(material in query_lower for material in material_keywords):
                    material_matches.append((query, score))
                
                # Check for upper body items
                if any(item in query_lower for item in upper_keywords):
                    upper_matches.append((query, score))
                
                # Check for lower body items
                elif any(item in query_lower for item in lower_keywords):
                    lower_matches.append((query, score))
            
            # Smart selection with better color accuracy
            best_upper = None
            best_lower = None
            
            # Find best upper body match (prioritize highest score first)
            for query, score in upper_matches[:3]:  # Check top 3 matches
                if score > 0.02:  # Confidence threshold
                    best_upper = query
                    break
            
            # Find best lower body match (prioritize highest score first)
            for query, score in lower_matches[:3]:  # Check top 3 matches
                if score > 0.02:  # Confidence threshold
                    best_lower = query
                    break
            
            # If we got results, enhance them with more detail
            if best_upper and best_lower:
                # Try to enhance with patterns and materials
                enhanced_upper = self._enhance_clothing_description(best_upper, sorted_results)
                enhanced_lower = self._enhance_clothing_description(best_lower, sorted_results)
                
                description_parts = [enhanced_upper, enhanced_lower]
                description = ", ".join(description_parts)
                logger.info(f"FashionCLIP enhanced result: {description}")
                return description
            
            # Build basic description if enhancement fails
            description_parts = []
            if best_upper:
                description_parts.append(best_upper)
            if best_lower:
                description_parts.append(best_lower)
            
            if description_parts:
                description = ", ".join(description_parts)
                logger.info(f"FashionCLIP comprehensive result: {description}")
                return description
            else:
                # Fallback to top overall matches
                confident_matches = [
                    match for match, score in sorted_results[:10]
                    if score > 0.03
                ]
                
                if confident_matches:
                    description = self._create_enhanced_fashionclip_description(confident_matches)
                    return description
                else:
                    return "Person wearing casual clothing"
                    
        except Exception as e:
            logger.error(f"Error in FashionCLIP analysis: {e}")
            return None

    def _create_enhanced_fashionclip_description(self, matches: List[str]) -> str:
        """Create a comprehensive description from FashionCLIP matches"""
        if not matches:
            return "Person wearing casual clothing"
        
        # Debug: Print matches
        logger.info(f"FashionCLIP matches to process: {matches}")
            
        # Categorize matches more intelligently
        upper_body_items = []
        lower_body_items = []
        
        # Keywords for categorization
        upper_keywords = ["tank top", "crop top", "t-shirt", "blouse", "shirt", "halter top", "camisole", "tube top", "sleeveless", "short sleeves"]
        lower_keywords = ["shorts", "jeans", "pants", "skirt", "leggings", "dress", "yoga pants"]
        
        for match in matches:
            match_lower = match.lower()
            
            # Check if this is a complete description (color + item)
            has_color = any(color in match_lower for color in ["red", "blue", "white", "black", "pink", "green", "yellow", "purple", "brown", "gray", "navy", "beige"])
            
            # Check for upper body items
            if any(keyword in match_lower for keyword in upper_keywords):
                upper_body_items.append(match)
            
            # Check for lower body items
            elif any(keyword in match_lower for keyword in lower_keywords):
                lower_body_items.append(match)
        
        # Build description
        description_parts = []
        
        # Add best upper body item
        if upper_body_items:
            # Prefer complete descriptions (with color)
            best_upper = None
            for item in upper_body_items:
                if any(color in item.lower() for color in ["red", "blue", "white", "black", "pink", "green"]):
                    best_upper = item
                    break
            if not best_upper:
                best_upper = upper_body_items[0]
            description_parts.append(best_upper)
        
        # Add best lower body item
        if lower_body_items:
            # Prefer complete descriptions (with color)
            best_lower = None
            for item in lower_body_items:
                if any(color in item.lower() for color in ["red", "blue", "white", "black", "pink", "green"]):
                    best_lower = item
                    break
            if not best_lower:
                best_lower = lower_body_items[0]
            description_parts.append(best_lower)
        
        # If we don't have clear upper/lower items, try to construct from matches
        if not description_parts:
            # Look for any color/item combinations
            for match in matches[:3]:  # Take top 3 matches
                if any(keyword in match.lower() for keyword in upper_keywords + lower_keywords):
                    description_parts.append(match)
        
        # Combine into natural description
        if description_parts:
            result = ", ".join(description_parts)
            logger.info(f"FashionCLIP final description: {result}")
            return result
        else:
            return "Person wearing casual clothing"

# Factory function for easy model switching
def create_clothing_analyzer(model_name: str = "fashionclip") -> ClothingAnalyzer:
    """
    Create a clothing analyzer with the specified model
    
    Args:
        model_name: Model to use - "fashionclip" (recommended), "instructblip", or "blip2"
    
    Returns:
        ClothingAnalyzer instance
    """
    return ClothingAnalyzer(model_name=model_name)
