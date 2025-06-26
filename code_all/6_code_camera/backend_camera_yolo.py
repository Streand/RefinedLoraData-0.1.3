"""
YOLO-based Camera Angle Detection Backend
Provides advanced pose detection for camera angle and framing analysis
Compatible with Blackwell NVIDIA GPUs (will auto-upgrade when PyTorch supports sm_120)
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class YOLOCameraAnalyzer:
    """
    Advanced camera angle analyzer using YOLO pose detection
    Provides more accurate pose analysis than OpenCV Haar cascades
    """
    
    def __init__(self, model_size: str = "nano"):
        """
        Initialize YOLO camera analyzer
        
        Args:
            model_size: YOLO model size ("nano", "small", "medium", "large", "extra_large")
        """
        self.model = None
        self.device = self._setup_device()
        self.model_size = model_size
        self.is_initialized = False
        self._load_model()
        
    def _setup_device(self) -> str:
        """
        Setup optimal device for inference with Blackwell GPU detection
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return 'cpu'
        
        try:
            # Check GPU compatibility
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            
            # Blackwell GPU detection
            if "RTX 50" in gpu_name or "B" in gpu_name or major >= 9:
                logger.info(f"Blackwell GPU detected: {gpu_name}")
                
                # Test if PyTorch supports Blackwell yet
                try:
                    test_tensor = torch.randn(1, device='cuda')
                    logger.info("✅ Blackwell GPU support available! Using GPU acceleration")
                    return 'cuda'
                except Exception as e:
                    if "kernel image" in str(e).lower() or "sm_120" in str(e):
                        logger.warning("⚠️  Blackwell GPU detected but not yet supported by PyTorch")
                        logger.warning("   Using CPU inference until PyTorch adds sm_120 support")
                        logger.info("   Performance will be significantly faster once GPU support is added")
                    else:
                        logger.warning(f"GPU error: {e}")
                    return 'cpu'
            else:
                # Non-Blackwell GPU
                logger.info(f"GPU detected: {gpu_name} (compute {major}.{minor})")
                return 'cuda'
                
        except Exception as e:
            logger.warning(f"GPU detection error: {e}")
            return 'cpu'
    
    def _load_model(self) -> bool:
        """
        Load YOLO pose detection model
        
        Returns:
            True if model loaded successfully
        """
        try:
            from ultralytics import YOLO
            
            # Model size mapping
            model_map = {
                "nano": "yolo11n-pose.pt",
                "small": "yolo11s-pose.pt", 
                "medium": "yolo11m-pose.pt",
                "large": "yolo11l-pose.pt",
                "extra_large": "yolo11x-pose.pt"
            }
            
            model_path = model_map.get(self.model_size, "yolo11n-pose.pt")
            
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            
            # Test model with dummy data
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_results = self.model(dummy_image, device=self.device, verbose=False)
            
            self.is_initialized = True
            logger.info(f"✅ YOLO model loaded successfully on {self.device}")
            
            return True
            
        except ImportError:
            logger.error("❌ Ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image for camera angle and pose information
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'YOLO model not initialized',
                'camera_angle': 'unknown',
                'framing': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'camera_angle': 'unknown',
                    'framing': 'unknown',
                    'confidence': 0.0
                }
            
            start_time = time.time()
            
            # Run YOLO inference
            results = self.model(image_path, device=self.device, verbose=False)
            
            inference_time = time.time() - start_time
            
            # Process results
            analysis = self._process_yolo_results(results)
            analysis['inference_time'] = inference_time
            analysis['device'] = self.device
            analysis['success'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'camera_angle': 'unknown',
                'framing': 'unknown',
                'confidence': 0.0
            }
    
    def _process_yolo_results(self, results) -> Dict:
        """
        Process YOLO detection results to determine camera angle and framing
        
        Args:
            results: YOLO detection results
            
        Returns:
            Dictionary with processed analysis
        """
        try:
            # Initialize default response
            analysis = {
                'camera_angle': 'unknown',
                'framing': 'unknown',
                'confidence': 0.0,
                'people_detected': 0,
                'keypoints_detected': False,
                'pose_analysis': {}
            }
            
            if not results or len(results) == 0:
                return analysis
            
            result = results[0]  # Use first result
            
            # Check for pose keypoints (PRIORITY: Use pose framing when available)
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidence = result.keypoints.conf.cpu().numpy()
                
                analysis['people_detected'] = len(keypoints)
                analysis['keypoints_detected'] = True
                
                if len(keypoints) > 0:
                    # Analyze primary person
                    person_analysis = self._analyze_person_pose(keypoints[0], confidence[0])
                    analysis.update(person_analysis)
                    
                    # IMPORTANT: Pose-based framing takes precedence over box-based
                    pose_framing = person_analysis.get('framing', 'unknown')
                    if pose_framing != 'unknown':
                        analysis['framing'] = pose_framing
                        logger.info(f"Using pose-based framing: {pose_framing}")
            
            # Box-based analysis (only as fallback when no pose framing available)
            use_box_framing = (analysis.get('framing', 'unknown') == 'unknown' and 
                              hasattr(result, 'boxes') and result.boxes is not None)
            
            if use_box_framing:
                boxes_tensor = result.boxes.xyxy.cpu().numpy()
                if len(boxes_tensor) > 0:
                    framing_analysis = self._analyze_framing_from_boxes(boxes_tensor, result.orig_img.shape)
                    box_framing = framing_analysis.get('framing', 'unknown')
                    if box_framing != 'unknown':
                        analysis['framing'] = box_framing
                        logger.info(f"Using box-based framing: {box_framing}")
                    if analysis['confidence'] == 0.0:
                        analysis['confidence'] = framing_analysis.get('confidence', 0.0)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing YOLO results: {e}")
            return {
                'camera_angle': 'unknown',
                'framing': 'unknown', 
                'confidence': 0.0,
                'people_detected': 0,
                'keypoints_detected': False,
                'error': str(e)
            }
    
    def _analyze_person_pose(self, keypoints: np.ndarray, confidence: np.ndarray) -> Dict:
        """
        Analyze person pose keypoints to determine camera angle
        
        Args:
            keypoints: Array of keypoint coordinates (17, 2)
            confidence: Array of keypoint confidence scores (17,)
            
        Returns:
            Dictionary with pose analysis
        """
        try:
            # COCO pose keypoints indices
            KEYPOINT_NAMES = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            visible_threshold = 0.5
            
            # Extract key points
            nose = keypoints[0] if confidence[0] > visible_threshold else None
            left_eye = keypoints[1] if confidence[1] > visible_threshold else None
            right_eye = keypoints[2] if confidence[2] > visible_threshold else None
            left_ear = keypoints[3] if confidence[3] > visible_threshold else None
            right_ear = keypoints[4] if confidence[4] > visible_threshold else None
            left_shoulder = keypoints[5] if confidence[5] > visible_threshold else None
            right_shoulder = keypoints[6] if confidence[6] > visible_threshold else None
            left_hip = keypoints[11] if confidence[11] > visible_threshold else None
            right_hip = keypoints[12] if confidence[12] > visible_threshold else None
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidence[confidence > visible_threshold])
            
            # Analyze pose symmetry and orientation
            camera_angle, framing = self._determine_angle_and_framing(
                nose, left_eye, right_eye, left_ear, right_ear,
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            
            return {
                'camera_angle': camera_angle,
                'framing': framing,
                'confidence': float(avg_confidence),
                'pose_analysis': {
                    'visible_keypoints': int(np.sum(confidence > visible_threshold)),
                    'symmetry_score': self._calculate_symmetry_score(keypoints, confidence),
                    'face_visibility': self._analyze_face_visibility(left_eye, right_eye, nose),
                    'body_orientation': self._analyze_body_orientation(left_shoulder, right_shoulder, left_hip, right_hip)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing person pose: {e}")
            return {
                'camera_angle': 'unknown',
                'framing': 'unknown',
                'confidence': 0.0
            }
    
    def _determine_angle_and_framing(self, nose, left_eye, right_eye, left_ear, right_ear,
                                   left_shoulder, right_shoulder, left_hip, right_hip) -> Tuple[str, str]:
        """
        Determine camera angle and framing based on visible keypoints
        
        Returns:
            Tuple of (camera_angle, framing)
        """
        try:
            # Analyze face orientation
            face_angle = "unknown"
            if left_eye is not None and right_eye is not None:
                # Both eyes visible - likely front facing or slight angle
                eye_distance = np.linalg.norm(left_eye - right_eye)
                if eye_distance > 15:  # Wide eye separation
                    face_angle = "front"
                elif eye_distance > 8:
                    face_angle = "slight_angle"
                else:
                    face_angle = "side"
            elif left_eye is not None or right_eye is not None:
                # Only one eye visible - likely profile
                if left_ear is not None or right_ear is not None:
                    face_angle = "profile"
                else:
                    face_angle = "partial_profile"
            
            # Analyze body symmetry
            body_symmetry = "unknown"
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
                if shoulder_distance > 30:
                    body_symmetry = "symmetric"
                elif shoulder_distance > 15:
                    body_symmetry = "partial_symmetric"
                else:
                    body_symmetry = "asymmetric"
            
            # Determine camera angle
            if face_angle == "front" and body_symmetry == "symmetric":
                camera_angle = "straight on"
            elif face_angle == "profile" or body_symmetry == "asymmetric":
                camera_angle = "side view"
            elif face_angle == "slight_angle" or body_symmetry == "partial_symmetric":
                camera_angle = "bilaterally symmetrical"
            else:
                camera_angle = "straight on"  # Default
            
            # Determine framing based on visible body parts
            framing = self._determine_framing_advanced(
                nose, left_eye, right_eye, left_ear, right_ear,
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            
            return camera_angle, framing
            
        except Exception as e:
            logger.error(f"Error determining angle and framing: {e}")
            return "unknown", "unknown"
    
    def _determine_framing(self, left_shoulder, right_shoulder, left_hip, right_hip) -> str:
        """
        Determine image framing based on visible body parts
        
        Returns:
            Framing type string
        """
        try:
            visible_shoulders = sum([1 for p in [left_shoulder, right_shoulder] if p is not None])
            visible_hips = sum([1 for p in [left_hip, right_hip] if p is not None])
            
            # More precise framing detection
            if visible_shoulders >= 1 and visible_hips >= 1:
                return "cowboy shot"  # Upper body + hips visible
            elif visible_shoulders >= 1:
                return "medium shot"  # Upper body visible
            else:
                return "close-up"  # Only face/head visible (no shoulders detected)
                
        except Exception as e:
            logger.error(f"Error determining framing: {e}")
            return "unknown"
    
    def _determine_framing_advanced(self, nose, left_eye, right_eye, left_ear, right_ear,
                                  left_shoulder, right_shoulder, left_hip, right_hip) -> str:
        """
        Advanced framing determination based on all visible keypoints
        
        Returns:
            Framing type string
        """
        try:
            # Count visible facial features
            face_points = [nose, left_eye, right_eye, left_ear, right_ear]
            visible_face = sum([1 for p in face_points if p is not None])
            
            # Count visible body parts
            visible_shoulders = sum([1 for p in [left_shoulder, right_shoulder] if p is not None])
            visible_hips = sum([1 for p in [left_hip, right_hip] if p is not None])
            
            # Advanced framing logic
            if visible_face >= 2 and visible_shoulders == 0 and visible_hips == 0:
                # Only face visible, no body parts
                return "close-up"
            elif visible_shoulders >= 1 and visible_hips >= 1:
                # Both shoulders and hips visible
                return "cowboy shot"
            elif visible_shoulders >= 1:
                # Shoulders visible but no hips
                return "medium shot"
            elif visible_face >= 1:
                # Only face features visible
                return "close-up"
            else:
                # Fallback to basic analysis
                return self._determine_framing(left_shoulder, right_shoulder, left_hip, right_hip)
                
        except Exception as e:
            logger.error(f"Error in advanced framing determination: {e}")
            return self._determine_framing(left_shoulder, right_shoulder, left_hip, right_hip)

    def _calculate_symmetry_score(self, keypoints: np.ndarray, confidence: np.ndarray) -> float:
        """Calculate pose symmetry score (0-1)"""
        try:
            # Pairs of symmetric keypoints
            symmetric_pairs = [
                (1, 2),   # eyes
                (3, 4),   # ears  
                (5, 6),   # shoulders
                (7, 8),   # elbows
                (9, 10),  # wrists
                (11, 12), # hips
                (13, 14), # knees
                (15, 16)  # ankles
            ]
            
            symmetry_scores = []
            for left_idx, right_idx in symmetric_pairs:
                if confidence[left_idx] > 0.5 and confidence[right_idx] > 0.5:
                    left_point = keypoints[left_idx]
                    right_point = keypoints[right_idx]
                    
                    # Calculate relative position symmetry
                    center_x = (left_point[0] + right_point[0]) / 2
                    left_dist = abs(left_point[0] - center_x)
                    right_dist = abs(right_point[0] - center_x) 
                    
                    if left_dist + right_dist > 0:
                        symmetry = 1 - abs(left_dist - right_dist) / (left_dist + right_dist)
                        symmetry_scores.append(symmetry)
            
            return float(np.mean(symmetry_scores)) if symmetry_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_face_visibility(self, left_eye, right_eye, nose) -> str:
        """Analyze face visibility level"""
        visible_count = sum([1 for p in [left_eye, right_eye, nose] if p is not None])
        
        if visible_count >= 3:
            return "full_face"
        elif visible_count >= 2:
            return "partial_face"
        elif visible_count >= 1:
            return "minimal_face"
        else:
            return "no_face"
    
    def _analyze_body_orientation(self, left_shoulder, right_shoulder, left_hip, right_hip) -> str:
        """Analyze body orientation"""
        shoulder_visible = sum([1 for p in [left_shoulder, right_shoulder] if p is not None])
        hip_visible = sum([1 for p in [left_hip, right_hip] if p is not None])
        
        if shoulder_visible >= 2 and hip_visible >= 2:
            return "frontal"
        elif shoulder_visible >= 1 and hip_visible >= 1:
            return "partial"
        else:
            return "profile"
    
    def _analyze_framing_from_boxes(self, boxes, image_shape) -> Dict:
        """
        Analyze framing based on detection bounding boxes
        
        Args:
            boxes: Detection bounding boxes in xyxy format (N, 4)
            image_shape: Original image dimensions
            
        Returns:
            Dictionary with framing analysis
        """
        try:
            if len(boxes) == 0:
                return {'framing': 'unknown', 'confidence': 0.0}
            
            # Ensure boxes is a 2D array
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            
            # Validate box format (should be N x 4 with [x1, y1, x2, y2])
            if boxes.shape[1] != 4:
                logger.warning(f"Invalid box format: {boxes.shape}, expected (N, 4)")
                return {'framing': 'unknown', 'confidence': 0.0}
            
            # Get largest detection (likely main subject)
            areas = []
            for i in range(len(boxes)):
                box = boxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height
                areas.append(area)
            
            main_box_idx = np.argmax(areas)
            main_box = boxes[main_box_idx]
            
            # Calculate box dimensions relative to image
            img_height, img_width = image_shape[:2]
            box_width = (main_box[2] - main_box[0]) / img_width
            box_height = (main_box[3] - main_box[1]) / img_height
            
            # Calculate aspect ratio of the detection box
            box_aspect_ratio = box_width / box_height if box_height > 0 else 1.0
            
            # Improved framing logic that considers box shape and size
            # For portrait/face shots, a large box doesn't mean full shot
            if box_height > 0.9 and box_width > 0.8:
                # Very large box covering most of image = full shot
                framing = "full shot"
            elif box_height > 0.7 and box_width > 0.6:
                # Large box but not covering everything = medium shot
                framing = "medium shot" 
            elif box_height > 0.5 and box_width > 0.4:
                # Medium-sized box = cowboy shot
                framing = "cowboy shot"
            elif box_aspect_ratio < 1.5:  # Tall/square box (likely portrait)
                # Portrait-shaped detection = close-up (face/head)
                framing = "close-up"
            else:
                # Wide box or small box = close-up
                framing = "close-up"
            
            confidence = min(box_width * box_height, 1.0)  # Size-based confidence
            
            return {'framing': framing, 'confidence': confidence}
            
        except Exception as e:
            logger.error(f"Error analyzing framing from boxes: {e}")
            logger.error(f"Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else 'unknown'}")
            logger.error(f"Boxes content: {boxes if len(str(boxes)) < 200 else 'too large to display'}")
            return {'framing': 'unknown', 'confidence': 0.0}
    
    def generate_stable_diffusion_prompt(self, analysis_result: Dict) -> str:
        """
        Generate Stable Diffusion prompt based on camera analysis
        
        Args:
            analysis_result: Result from analyze_image()
            
        Returns:
            Formatted prompt string for Stable Diffusion
        """
        try:
            if not analysis_result.get('success', False):
                return "portrait, medium shot, straight on"
            
            camera_angle = analysis_result.get('camera_angle', 'straight on')
            framing = analysis_result.get('framing', 'medium shot')
            confidence = analysis_result.get('confidence', 0.0)
            
            # Map camera angles to SD terms
            angle_mapping = {
                'straight on': 'facing viewer, front view',
                'side view': 'profile view, side angle',
                'bilaterally symmetrical': 'three-quarter view, slight angle',
                'unknown': 'portrait'
            }
            
            # Map framing to SD terms
            framing_mapping = {
                'close-up': 'close-up, head and shoulders',
                'medium shot': 'medium shot, upper body',
                'cowboy shot': 'cowboy shot, waist up',
                'full shot': 'full body shot',
                'unknown': 'medium shot'
            }
            
            angle_prompt = angle_mapping.get(camera_angle, 'portrait')
            framing_prompt = framing_mapping.get(framing, 'medium shot')
            
            # Combine prompts
            prompt_parts = [framing_prompt, angle_prompt]
            
            # Add confidence-based modifiers
            if confidence > 0.8:
                prompt_parts.append("high detail")
            elif confidence > 0.6:
                prompt_parts.append("detailed")
            
            return ", ".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error generating SD prompt: {e}")
            return "portrait, medium shot, straight on"
    
    def get_device_info(self) -> Dict:
        """
        Get information about the current device and YOLO model status
        
        Returns:
            Dictionary with device and model information
        """
        info = {
            'device': self.device,
            'model_initialized': self.is_initialized,
            'model_size': self.model_size,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            try:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                major, minor = torch.cuda.get_device_capability(0)
                info['compute_capability'] = f"{major}.{minor}"
                
                # Blackwell detection
                gpu_name = info['gpu_name']
                if "RTX 50" in gpu_name or "B" in gpu_name or major >= 9:
                    info['is_blackwell'] = True
                    info['blackwell_support'] = self.device == 'cuda'
                else:
                    info['is_blackwell'] = False
                    info['blackwell_support'] = None
                    
            except Exception as e:
                info['gpu_error'] = str(e)
        
        return info
    
    def get_stable_diffusion_prompt(self, analysis_result: Dict) -> str:
        """
        Compatibility method for main UI integration
        Alias for generate_stable_diffusion_prompt()
        
        Args:
            analysis_result: Result from analyze_image()
            
        Returns:
            Formatted prompt string for Stable Diffusion
        """
        return self.generate_stable_diffusion_prompt(analysis_result)


# Convenience function for easy import
def create_yolo_analyzer(model_size: str = "nano") -> YOLOCameraAnalyzer:
    """
    Create a YOLO camera analyzer instance
    
    Args:
        model_size: YOLO model size ("nano", "small", "medium", "large", "extra_large")
        
    Returns:
        YOLOCameraAnalyzer instance
    """
    return YOLOCameraAnalyzer(model_size=model_size)


if __name__ == "__main__":
    # Test the YOLO analyzer
    analyzer = YOLOCameraAnalyzer()
    device_info = analyzer.get_device_info()
    
    print("🔍 YOLO Camera Analyzer Device Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    if analyzer.is_initialized:
        print("✅ YOLO analyzer ready for production use!")
    else:
        print("❌ YOLO analyzer initialization failed")
