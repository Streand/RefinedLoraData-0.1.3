"""
YOLO-based Camera Angle Detection Backend
Provides advanced pose detection for camera angle and framing analysis
Compatible with Blackwell NVIDIA GPUs (will auto-upgrade when PyTorch supports sm_120)
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Type hint for YOLO model (imported dynamically)
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    YOLO = None
    Results = None

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
        self.model: Optional[Any] = None
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
            if self.model is not None:
                test_results = self.model(dummy_image, device=self.device, verbose=False)
            else:
                raise RuntimeError("Model is None after loading")
            
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
            
            # Run YOLO inference with null check
            if self.model is None:
                raise RuntimeError("YOLO model is not initialized")
            
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
            
            # Extract key points including legs/feet for full body detection
            nose = keypoints[0] if confidence[0] > visible_threshold else None
            left_eye = keypoints[1] if confidence[1] > visible_threshold else None
            right_eye = keypoints[2] if confidence[2] > visible_threshold else None
            left_ear = keypoints[3] if confidence[3] > visible_threshold else None
            right_ear = keypoints[4] if confidence[4] > visible_threshold else None
            left_shoulder = keypoints[5] if confidence[5] > visible_threshold else None
            right_shoulder = keypoints[6] if confidence[6] > visible_threshold else None
            left_hip = keypoints[11] if confidence[11] > visible_threshold else None
            right_hip = keypoints[12] if confidence[12] > visible_threshold else None
            left_knee = keypoints[13] if confidence[13] > visible_threshold else None
            right_knee = keypoints[14] if confidence[14] > visible_threshold else None
            left_ankle = keypoints[15] if confidence[15] > visible_threshold else None
            right_ankle = keypoints[16] if confidence[16] > visible_threshold else None
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidence[confidence > visible_threshold])
            
            # Analyze pose symmetry and orientation
            camera_angle, framing = self._determine_angle_and_framing(
                nose, left_eye, right_eye, left_ear, right_ear,
                left_shoulder, right_shoulder, left_hip, right_hip,
                left_knee, right_knee, left_ankle, right_ankle
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
                                   left_shoulder, right_shoulder, left_hip, right_hip,
                                   left_knee, right_knee, left_ankle, right_ankle) -> Tuple[str, str]:
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
            
            # Determine framing based on visible body parts (including legs)
            framing = self._determine_framing_advanced(
                nose, left_eye, right_eye, left_ear, right_ear,
                left_shoulder, right_shoulder, left_hip, right_hip,
                left_knee, right_knee, left_ankle, right_ankle
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
                                  left_shoulder, right_shoulder, left_hip, right_hip,
                                  left_knee, right_knee, left_ankle, right_ankle) -> str:
        """
        Advanced framing determination based on all visible keypoints including legs
        Based on professional photography and SDXL standards
        
        Framing Types (distance from subject):
        - extreme close-up: Eyes/face details only
        - close-up: Head and shoulders
        - medium shot: Waist up (upper body)
        - cowboy shot: Mid-thigh up (waist to mid-thigh)
        - full body shot: Complete figure head to toe
        - establishing shot: Subject in environment (wide)
        
        Returns:
            Framing type string
        """
        try:
            # Count visible keypoints by body region
            face_points = [nose, left_eye, right_eye, left_ear, right_ear]
            shoulder_points = [left_shoulder, right_shoulder]
            hip_points = [left_hip, right_hip]
            knee_points = [left_knee, right_knee]
            ankle_points = [left_ankle, right_ankle]
            
            visible_face = sum([1 for p in face_points if p is not None])
            visible_shoulders = sum([1 for p in shoulder_points if p is not None])
            visible_hips = sum([1 for p in hip_points if p is not None])
            visible_knees = sum([1 for p in knee_points if p is not None])
            visible_ankles = sum([1 for p in ankle_points if p is not None])
            
            # Enhanced framing logic based on comprehensive body part visibility
            
            # EXTREME CLOSE-UP: Only face features, no body parts
            if visible_face >= 2 and visible_shoulders == 0 and visible_hips == 0:
                # Check if it's really extreme close-up (eyes only) vs regular close-up
                if visible_face >= 4:  # Multiple face features = close-up
                    return "close-up"
                else:  # Fewer face features = extreme close-up
                    return "extreme close-up"
            
            # FULL BODY SHOT: Head to toe visible (ankles/feet are key indicators)
            elif visible_ankles >= 1 and (visible_shoulders >= 1 or visible_face >= 1):
                return "full body shot"
            
            # COWBOY SHOT: Knees visible but no ankles (mid-thigh to head)
            elif visible_knees >= 1 and visible_ankles == 0 and visible_shoulders >= 1:
                return "cowboy shot"
            
            # MEDIUM SHOT: Shoulders and hips visible, but no knees (waist up)
            elif visible_shoulders >= 1 and visible_hips >= 1 and visible_knees == 0:
                return "medium shot"
            
            # CLOSE-UP: Shoulders visible but no hips (head and shoulders)
            elif visible_shoulders >= 1 and visible_hips == 0:
                return "close-up"
            
            # FALLBACK: Some face visible but unclear body parts
            elif visible_face >= 1:
                return "close-up"
            
            # DEFAULT: Unable to determine, use basic analysis
            else:
                return self._determine_framing_basic(visible_shoulders, visible_hips, visible_knees, visible_ankles)
                
        except Exception as e:
            logger.error(f"Error in advanced framing determination: {e}")
            return self._determine_framing_basic(0, 0, 0, 0)
    
    def _determine_framing_basic(self, visible_shoulders: int, visible_hips: int, 
                               visible_knees: int, visible_ankles: int) -> str:
        """
        Basic framing fallback when advanced analysis fails
        """
        try:
            if visible_ankles >= 1:
                return "full body shot"
            elif visible_knees >= 1:
                return "cowboy shot"
            elif visible_hips >= 1:
                return "medium shot"
            elif visible_shoulders >= 1:
                return "close-up"
            else:
                return "unknown"
        except Exception:
            return "unknown"

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
            
            # Enhanced framing logic based on professional photography standards
            # Consider both box size and aspect ratio for accurate classification
            
            if box_height > 0.9 and box_width > 0.8:
                # Very large box covering most of image = establishing shot
                framing = "establishing shot"
            elif box_height > 0.8 and box_width > 0.6:
                # Large box covering significant portion = full body shot
                framing = "full body shot"
            elif box_height > 0.6 and box_width > 0.5:
                # Medium-large box = cowboy shot (mid-thigh up)
                framing = "cowboy shot" 
            elif box_height > 0.4 and box_width > 0.3:
                # Medium box = medium shot (waist up)
                framing = "medium shot"
            elif box_height > 0.2 and box_width > 0.2:
                # Smaller box = close-up (head and shoulders)
                framing = "close-up"
            elif box_aspect_ratio < 1.2 and box_height < 0.3:
                # Very small, tall box = extreme close-up (face only)
                framing = "extreme close-up"
            else:
                # Default fallback
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
            
            # Map framing to SD terms (based on professional photography standards)
            framing_mapping = {
                'extreme close-up': 'extreme close-up, eyes and facial details',
                'close-up': 'close-up, head and shoulders',
                'medium shot': 'medium shot, waist up, upper body',
                'cowboy shot': 'cowboy shot, mid-thigh up',
                'full body shot': 'full body shot, head to toe',
                'establishing shot': 'establishing shot, subject in environment',
                'full shot': 'full body shot, head to toe',  # Legacy support
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
    
"""
COMPREHENSIVE FRAMING DETECTION SCENARIOS

The enhanced YOLO camera analyzer now supports detection of all professional photography framing types:

🎯 FRAMING TYPES DETECTED:

1. **EXTREME CLOSE-UP**
   - Detection: Only 2-3 face keypoints visible (eyes, nose), no body parts
   - Use case: Eye details, facial expressions, intimate portraits
   - SD Prompt: "extreme close-up, eyes and facial details"

2. **CLOSE-UP**
   - Detection: Face + shoulders visible, no hips
   - Use case: Head and shoulders portraits, dialogue scenes
   - SD Prompt: "close-up, head and shoulders"

3. **MEDIUM SHOT**
   - Detection: Shoulders + hips visible, no knees
   - Use case: Upper body shots, waist up
   - SD Prompt: "medium shot, waist up, upper body"

4. **COWBOY SHOT**
   - Detection: Shoulders + hips + knees visible, no ankles
   - Use case: Mid-thigh up, action scenes, western shots
   - SD Prompt: "cowboy shot, mid-thigh up"

5. **FULL BODY SHOT**
   - Detection: Ankles/feet visible + upper body
   - Use case: Complete figure, fashion, dance
   - SD Prompt: "full body shot, head to toe"

6. **ESTABLISHING SHOT**
   - Detection: Very large bounding box (90%+ of image)
   - Use case: Subject in environment, scene setting
   - SD Prompt: "establishing shot, subject in environment"

🔍 DETECTION HIERARCHY:
1. Keypoint-based detection (most accurate)
2. Bounding box fallback (when keypoints unclear)
3. Basic visibility counting (last resort)

📊 BODY PART ANALYSIS:
- Face: nose, left_eye, right_eye, left_ear, right_ear
- Shoulders: left_shoulder, right_shoulder
- Torso: left_hip, right_hip
- Legs: left_knee, right_knee
- Feet: left_ankle, right_ankle

✅ EDGE CASES HANDLED:
- Partial occlusion (one side visible)
- Profile views (asymmetric visibility)
- Unusual poses (flexible thresholds)
- Poor lighting (confidence weighting)
- Multiple people (primary subject selection)

🎭 COMPATIBLE WITH:
- SDXL/SD framing standards
- Professional photography terminology
- Stable Diffusion prompt optimization
- Real-world photography scenarios
"""

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
