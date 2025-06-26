"""
Camera analysis module
Contains camera angle and framing analysis with OpenCV and YOLO backends
"""

try:
    from .backend_camera import CameraAnalyzer
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    CameraAnalyzer = None

try:
    from .backend_camera_yolo import YOLOCameraAnalyzer
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOCameraAnalyzer = None

__all__ = ['CameraAnalyzer', 'YOLOCameraAnalyzer', 'OPENCV_AVAILABLE', 'YOLO_AVAILABLE']
