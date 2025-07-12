"""
Quick Fix for InstructBLIP Import Issue
This file provides a simple workaround for the model loading error
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch
    import PIL.Image as Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    torch_available = True
    print("✓ Basic imports successful")
except ImportError as e:
    torch_available = False
    print(f"✗ Import error: {e}")

class SimpleClothingAnalyzer:
    """Simplified clothing analyzer using only BLIP-2 to avoid InstructBLIP issues"""
    
    def __init__(self):
        """Initialize with BLIP-2 only for now"""
        self.device = self._setup_device()
        self.model = None
        self.processor = None
        self.is_initialized = False
        self.model_name = "blip2"
        
        # Load BLIP-2 model
        self._load_blip2_model()
    
    def _setup_device(self) -> str:
        """Simple device setup"""
        if not torch_available:
            return 'cpu'
        
        if torch.cuda.is_available():
            print("✓ CUDA available, using GPU")
            return 'cuda'
        else:
            print("! CUDA not available, using CPU")
            return 'cpu'
    
    def _load_blip2_model(self) -> bool:
        """Load BLIP-2 model only"""
        if not torch_available:
            print("✗ PyTorch not available")
            return False
        
        try:
            print("Loading BLIP-2 model...")
            model_id = "Salesforce/blip2-opt-2.7b"
            
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == 'cuda':
                self.model = self.model.to('cuda')
            
            self.is_initialized = True
            print("✓ BLIP-2 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load BLIP-2: {e}")
            return False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for clothing"""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create prompt for clothing analysis
            prompt = "Question: What clothing and accessories is this person wearing? Answer:"
            
            # Process image
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate description
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    num_beams=3
                )
            
            # Decode response
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Create simple categorization
            categorized = self._simple_categorize(response)
            
            # Create SD prompt
            sd_prompt = self._create_sd_prompt(response)
            
            return {
                "raw_description": response,
                "categorized": categorized,
                "sd_prompt": sd_prompt,
                "confidence": 0.85,  # Fixed confidence for now
                "model_used": "blip2",
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _simple_categorize(self, description: str) -> Dict[str, List[str]]:
        """Simple categorization based on keywords"""
        desc_lower = description.lower()
        
        categories = {
            "upper_body": [],
            "lower_body": [],
            "footwear": [],
            "accessories": []
        }
        
        # Simple keyword matching
        upper_keywords = ["shirt", "t-shirt", "blouse", "sweater", "jacket", "blazer", "coat", "top"]
        lower_keywords = ["pants", "jeans", "trousers", "shorts", "skirt", "dress"]
        footwear_keywords = ["shoes", "sneakers", "boots", "sandals", "heels"]
        accessory_keywords = ["hat", "bag", "jewelry", "watch", "glasses", "belt"]
        
        for keyword in upper_keywords:
            if keyword in desc_lower:
                categories["upper_body"].append(keyword)
        
        for keyword in lower_keywords:
            if keyword in desc_lower:
                categories["lower_body"].append(keyword)
                
        for keyword in footwear_keywords:
            if keyword in desc_lower:
                categories["footwear"].append(keyword)
                
        for keyword in accessory_keywords:
            if keyword in desc_lower:
                categories["accessories"].append(keyword)
        
        return categories
    
    def _create_sd_prompt(self, description: str) -> str:
        """Create SD-friendly prompt"""
        # Simple cleanup and formatting
        sd_prompt = description.replace("The person is wearing", "").strip()
        sd_prompt = sd_prompt.replace("They are wearing", "").strip()
        sd_prompt = sd_prompt.replace(".", ", ").replace("  ", " ")
        
        if not sd_prompt.endswith(", "):
            sd_prompt += ", "
        
        sd_prompt += "detailed clothing, high quality"
        
        return sd_prompt
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {
            "device": self.device,
            "model_name": self.model_name,
            "model_initialized": self.is_initialized,
            "torch_available": torch_available
        }
        
        if torch_available:
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name()
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info

def create_clothing_analyzer(model_name: str = "blip2"):
    """Create clothing analyzer - only BLIP-2 for now"""
    if model_name == "instructblip":
        print("⚠️ InstructBLIP temporarily disabled due to import issues")
        print("🔄 Using BLIP-2 instead")
    
    return SimpleClothingAnalyzer()

# Test the fix
if __name__ == "__main__":
    print("Testing simple clothing analyzer...")
    try:
        analyzer = create_clothing_analyzer()
        print("✓ Analyzer created successfully")
        
        # Test with a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='white')
        dummy_image.save("test_image.jpg")
        
        result = analyzer.analyze_image("test_image.jpg")
        if "error" not in result:
            print("✓ Analysis test successful")
        else:
            print(f"✗ Analysis test failed: {result['error']}")
        
        # Cleanup
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
