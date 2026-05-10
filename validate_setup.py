#!/usr/bin/env python3
"""
Validation script for YOLOv8 Live Object Detection & Tracking

This script checks if all dependencies are installed and the system is ready to run.
"""

import sys
import importlib
import subprocess
import cv2

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("[FAIL] Python 3.8 or higher is required")
        return False
    print(f"[OK] Python {sys.version.split()[0]} is compatible")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'flask',
        'ultralytics',
        'cv2',
        'deep_sort_realtime',
        'numpy',
        'torch',
        'torchvision'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[FAIL] Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_camera():
    """Check if camera is accessible"""
    print("\nChecking camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[FAIL] Cannot access camera (index 0)")
            print("Try different camera indices or check camera permissions")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[FAIL] Cannot read from camera")
            return False
        
        print("[OK] Camera is accessible")
        return True
        
    except Exception as e:
        print(f"[FAIL] Camera error: {e}")
        return False

def check_yolo_model():
    """Check if YOLOv8 model can be loaded"""
    print("\nChecking YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("[OK] YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] YOLOv8 model error: {e}")
        return False

def check_project_files():
    """Check if all project files exist"""
    print("\nChecking project files...")
    
    required_files = [
        'app.py',
        'detect_and_track.py',
        'requirements.txt',
        'config.py',
        'templates/index.html',
        'static/style.css'
    ]
    
    import os
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[FAIL] {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n[FAIL] Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def main():
    """Run all validation checks"""
    print("YOLOv8 Live Object Detection & Tracking - System Validation")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Files", check_project_files),
        ("Camera Access", check_camera),
        ("YOLOv8 Model", check_yolo_model)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n[{passed + 1}/{total}] {check_name}")
        print("-" * 40)
        if check_func():
            passed += 1
        else:
            print(f"\n[WARNING]  {check_name} check failed!")
    
    print("\n" + "=" * 70)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("[SUCCESS] All checks passed! Your system is ready to run the application.")
        print("\nTo start the application:")
        print("  python app.py")
        print("  or")
        print("  python run.py")
        print("\nThen open: http://localhost:5000")
    else:
        print("[FAIL] Some checks failed. Please fix the issues above before running.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check camera permissions")
        print("  - Ensure Python 3.8+ is installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
