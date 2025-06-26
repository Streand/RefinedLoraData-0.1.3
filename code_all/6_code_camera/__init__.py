"""
Camera analysis module
Contains YOLO-based camera angle and framing analysis
"""

try:
    from .backend_camera_yolo import YOLOCameraAnalyzer
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOCameraAnalyzer = None

__all__ = ['YOLOCameraAnalyzer', 'YOLO_AVAILABLE']
