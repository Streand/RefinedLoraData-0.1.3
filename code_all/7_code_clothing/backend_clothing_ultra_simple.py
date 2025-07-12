"""
Ultra-Simple Clothing Analyzer using BLIP base model
This bypasses the complex model loading issues
"""

import os
import time
import json
from typing import Dict, Any, Optional, List

try:
    import torch
    import PIL.Image as Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    torch_available = True
    print("✓ Imports successful")
except ImportError as e:
    torch_available = False
    print(f"✗ Import error: {e}")

class UltraSimpleClothingAnalyzer:
    """Ultra-simple clothing analyzer using base BLIP model"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.is_initialized = False
        self.model_name = "blip2"
        
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the simplest BLIP model"""
        try:
            print("Loading BLIP base model...")
            
            # Use the original BLIP model which is more stable
            model_id = "Salesforce/blip-image-captioning-base"
            
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id)
            
            if self.device == 'cuda':
                self.model = self.model.to('cuda')
            
            self.is_initialized = True
            print("✓ BLIP model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.is_initialized = False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for clothing"""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Use conditional generation for clothing
            prompt = "a photo of clothing, "
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate description
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50)
            
            # Decode
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up
            if prompt in caption:
                caption = caption.replace(prompt, "").strip()
            
            # Simple clothing extraction
            clothing_desc = self._extract_clothing_terms(caption)
            
            return {
                "raw_description": clothing_desc,
                "categorized": {"general": [clothing_desc]},
                "sd_prompt": f"{clothing_desc}, detailed clothing, high quality",
                "confidence": 0.8,
                "model_used": "blip_base",
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_clothing_terms(self, caption: str) -> str:
        """Extract clothing-related terms"""
        # Simple approach: look for clothing keywords
        clothing_keywords = [
            "shirt", "dress", "pants", "jeans", "jacket", "coat", "sweater",
            "skirt", "shoes", "boots", "hat", "clothing", "outfit", "wearing"
        ]
        
        caption_lower = caption.lower()
        found_terms = []
        
        for keyword in clothing_keywords:
            if keyword in caption_lower:
                found_terms.append(keyword)
        
        if found_terms:
            return f"person wearing {', '.join(found_terms)}"
        else:
            return caption  # Return original if no clothing terms found
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device info"""
        return {
            "device": self.device,
            "model_name": "blip_base",
            "model_initialized": self.is_initialized,
            "torch_available": torch_available,
            "cuda_available": torch.cuda.is_available() if torch_available else False
        }

def create_clothing_analyzer(model_name: str = "blip2"):
    """Create the ultra-simple analyzer"""
    print("🔧 Creating ultra-simple clothing analyzer...")
    return UltraSimpleClothingAnalyzer()

# Test
if __name__ == "__main__":
    print("Testing ultra-simple analyzer...")
    try:
        analyzer = create_clothing_analyzer()
        if analyzer.is_initialized:
            print("✓ Model ready for use")
        else:
            print("✗ Model failed to initialize")
    except Exception as e:
        print(f"✗ Test failed: {e}")
