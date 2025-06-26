"""
Backend for Camera Angle and Framing Analysis
Analyzes images to determine camera framing and angle information for Stable Diffusion prompts
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraAnalyzer:
    """Analyzes camera angles and framing in images"""
    
    def __init__(self):
        self.framing_keywords = [
            "extreme close-up",
            "close-up", 
            "medium shot",
            "full body shot",
            "establishing shot"
        ]
        
        self.angle_keywords = [
            "straight on",
            "bilaterally symmetrical",
            "side view",
            "back view",
            "from above",
            "from below", 
            "wide angle view",
            "fisheye view",
            "overhead shot",
            "top down shot",
            "hero view",
            "selfie"
        ]
        
        # Load face cascade for face detection
        try:
            import cv2
            import os
            # Try multiple ways to get the cascade files
            cascade_paths = []
            
            # Method 1: Using cv2.data (newer versions)
            try:
                cascade_paths.append(getattr(cv2, 'data').haarcascades)
            except AttributeError:
                pass
            
            # Method 2: Using cv2 installation path
            try:
                cv2_dir = os.path.dirname(cv2.__file__)
                cascade_paths.append(os.path.join(cv2_dir, 'data'))
            except:
                pass
            
            # Method 3: Common system paths
            cascade_paths.extend([
                '/usr/share/opencv4/haarcascades/',
                '/usr/local/share/opencv/haarcascades/',
                'C:/ProgramData/Anaconda3/Library/etc/haarcascades/',
            ])
            
            self.face_cascade = None
            self.profile_cascade = None
            
            for path in cascade_paths:
                try:
                    if os.path.exists(os.path.join(path, 'haarcascade_frontalface_default.xml')):
                        self.face_cascade = cv2.CascadeClassifier(os.path.join(path, 'haarcascade_frontalface_default.xml'))
                        self.profile_cascade = cv2.CascadeClassifier(os.path.join(path, 'haarcascade_profileface.xml'))
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not load face cascades: {e}")
            self.face_cascade = None
            self.profile_cascade = None

    def analyze_image(self, image_path: str) -> Dict[str, str]:
        """
        Analyze an image to determine camera framing and angle
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with framing and angle analysis
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = image.shape[:2]
            
            # Analyze framing
            framing = self._analyze_framing(image, gray)
            
            # Analyze angle
            angle = self._analyze_angle(image, gray)
            
            # Additional analysis
            aspect_ratio = self._get_aspect_ratio(width, height)
            composition_notes = self._analyze_composition(image, gray)
            
            return {
                "framing": framing,
                "angle": angle,
                "aspect_ratio": aspect_ratio,
                "composition_notes": composition_notes,
                "image_dimensions": f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}

    def _analyze_framing(self, image: np.ndarray, gray: np.ndarray) -> str:
        """Analyze the framing/distance of the shot"""
        height, width = gray.shape
        
        # Detect faces to help determine framing
        faces = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Calculate face size relative to image
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_area = largest_face[2] * largest_face[3]
            image_area = width * height
            face_ratio = face_area / image_area
            
            # Determine framing based on face size
            if face_ratio > 0.3:
                return "extreme close-up"
            elif face_ratio > 0.15:
                return "close-up"
            elif face_ratio > 0.05:
                return "medium shot"
            elif face_ratio > 0.01:
                return "full body shot"
            else:
                return "establishing shot"
        else:
            # No faces detected - analyze based on image content
            # Use edge detection to estimate subject size
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (presumably the main subject)
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                image_area = width * height
                subject_ratio = contour_area / image_area
                
                if subject_ratio > 0.4:
                    return "close-up"
                elif subject_ratio > 0.2:
                    return "medium shot"
                elif subject_ratio > 0.1:
                    return "full body shot"
                else:
                    return "establishing shot"
            
        return "medium shot"  # Default fallback

    def _analyze_angle(self, image: np.ndarray, gray: np.ndarray) -> str:
        """Analyze the camera angle/position"""
        height, width = gray.shape
        
        # Detect faces for angle analysis
        front_faces = []
        profile_faces = []
        
        if self.face_cascade is not None:
            front_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
        if self.profile_cascade is not None:
            profile_faces = self.profile_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Analyze based on face detection
        if len(front_faces) > 0:
            face = front_faces[0]
            face_center_y = face[1] + face[3] // 2
            image_center_y = height // 2
            
            # Check if face is centered (bilateral symmetry)
            face_center_x = face[0] + face[2] // 2
            image_center_x = width // 2
            
            if abs(face_center_x - image_center_x) < width * 0.1:
                if abs(face_center_y - image_center_y) < height * 0.1:
                    return "bilaterally symmetrical"
                elif face_center_y < image_center_y * 0.7:
                    return "from below"
                elif face_center_y > image_center_y * 1.3:
                    return "from above"
                else:
                    return "straight on"
            else:
                return "straight on"
                
        elif len(profile_faces) > 0:
            return "side view"
        
        # Analyze perspective using horizon detection
        angle = self._detect_perspective(gray)
        
        if angle:
            return angle
            
        return "straight on"  # Default fallback

    def _detect_perspective(self, gray: np.ndarray) -> Optional[str]:
        """Detect perspective using line detection and vanishing points"""
        # Use HoughLines to detect perspective lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 0:
            # Analyze line orientations to determine perspective
            vertical_lines = 0
            horizontal_lines = 0
            diagonal_lines = 0
            
            for line in lines:
                try:
                    # Handle different line format possibilities
                    line_data = np.array(line).flatten()
                    if len(line_data) >= 2:
                        rho, theta = line_data[0], line_data[1]
                        angle = float(theta) * 180 / np.pi
                        
                        if 80 <= angle <= 100:  # Near vertical
                            vertical_lines += 1
                        elif angle <= 10 or angle >= 170:  # Near horizontal
                            horizontal_lines += 1
                        else:
                            diagonal_lines += 1
                except:
                    continue
            
            # Determine perspective based on line distribution
            if diagonal_lines > vertical_lines + horizontal_lines:
                return "wide angle view"
            elif vertical_lines > horizontal_lines * 2:
                return "from below"
            elif horizontal_lines > vertical_lines * 2:
                return "from above"
        
        return None

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Determine aspect ratio category"""
        ratio = width / height
        
        if ratio > 1.5:
            return "wide landscape"
        elif ratio > 1.1:
            return "landscape"
        elif ratio > 0.9:
            return "square"
        elif ratio > 0.7:
            return "portrait"
        else:
            return "tall portrait"

    def _analyze_composition(self, image: np.ndarray, gray: np.ndarray) -> str:
        """Analyze additional composition elements"""
        height, width = gray.shape
        notes = []
        
        # Check for central composition
        center_region = gray[height//3:2*height//3, width//3:2*width//3]
        center_mean = np.mean(center_region)
        overall_mean = np.mean(gray)
        
        if center_mean > overall_mean * 1.2:
            notes.append("centered composition")
        
        # Check for symmetry
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        if right_half.shape == left_half.shape:
            symmetry_diff = np.mean(np.abs(left_half - right_half))
            if symmetry_diff < 30:
                notes.append("symmetrical")
        
        # Check for rule of thirds
        third_h = height // 3
        third_w = width // 3
        
        # Check interest points at rule of thirds intersections
        interest_points = [
            gray[third_h, third_w],
            gray[third_h, 2*third_w],
            gray[2*third_h, third_w],
            gray[2*third_h, 2*third_w]
        ]
        
        if max([float(p) for p in interest_points]) > overall_mean * 1.3:
            notes.append("rule of thirds")
        
        return ", ".join(notes) if notes else "standard composition"

    def analyze_batch(self, image_folder: str) -> Dict[str, Dict]:
        """
        Analyze multiple images in a folder
        
        Args:
            image_folder: Path to folder containing images
            
        Returns:
            Dictionary with analysis results for each image
        """
        results = {}
        
        if not os.path.exists(image_folder):
            return {"error": {"error": "Folder does not exist"}}
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [f for f in os.listdir(image_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            results[image_file] = self.analyze_image(image_path)
            
        return results

    def get_stable_diffusion_prompt(self, analysis: Dict[str, str]) -> str:
        """
        Generate Stable Diffusion prompt components based on analysis
        
        Args:
            analysis: Analysis results from analyze_image
            
        Returns:
            String with camera-related prompt components
        """
        prompt_parts = []
        
        if "framing" in analysis and analysis["framing"]:
            prompt_parts.append(f"({analysis['framing']}:1.2)")
            
        if "angle" in analysis and analysis["angle"]:
            prompt_parts.append(f"({analysis['angle']}:1.1)")
            
        return ", ".join(prompt_parts)


# Example usage and testing functions
def test_analyzer():
    """Test the camera analyzer with sample images"""
    analyzer = CameraAnalyzer()
    
    # Test with a sample image path
    test_image = "sample.jpg"  # Replace with actual test image
    
    if os.path.exists(test_image):
        result = analyzer.analyze_image(test_image)
        print("Analysis Result:", result)
        
        prompt = analyzer.get_stable_diffusion_prompt(result)
        print("SD Prompt Components:", prompt)
    else:
        print("No test image found")


if __name__ == "__main__":
    test_analyzer()