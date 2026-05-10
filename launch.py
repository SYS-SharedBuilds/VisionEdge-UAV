#!/usr/bin/env python3
"""
Simple launcher script for YOLOv8 Live Object Detection & Tracking

This script provides an easy way to start the application with validation.
"""

import os
# Fix OpenMP runtime conflict - must be set before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import subprocess

def run_validation():
    """Run the validation script"""
    print("Running system validation...")
    try:
        result = subprocess.run([sys.executable, 'validate_setup.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Validation warnings/errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running validation: {e}")
        return False

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies...")
    
    # Check Python version and use appropriate requirements file
    python_version = sys.version_info
    print(f"Detected Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    requirements_file = 'requirements.txt'
    if python_version >= (3, 13):
        if os.path.exists('requirements-python313.txt'):
            print("Using Python 3.13+ optimized requirements...")
            requirements_file = 'requirements-python313.txt'
    
    try:
        # Try installing with upgraded pip first
        print("Upgrading pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        print(f"Installing from {requirements_file}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], 
                      check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nTrying alternative installation method...")
        
        # Try installing packages individually with more flexibility
        return install_packages_individually()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def install_packages_individually():
    """Install packages one by one with fallback versions"""
    packages = [
        'flask',
        'ultralytics',
        'opencv-python',
        'numpy',
        'pillow',
        'torch',
        'torchvision',
        'deep-sort-realtime',
        'scipy',
        'msgpack-rpc-python',
        'airsim'
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                          check=True, capture_output=True)
            print(f"[OK] {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"[FAIL] Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\nFailed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually.")
        return len(failed_packages) == 0
    
    return True

def start_application():
    """Start the main application"""
    print("Starting YOLOv8 Live Object Detection & Tracking...")
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting application: {e}")

def main():
    """Main launcher function"""
    print("YOLOv8 Live Object Detection & Tracking - Launcher")
    print("=" * 60)
    
    # Check if validation script exists
    if not os.path.exists('validate_setup.py'):
        print("Warning: validate_setup.py not found. Skipping validation.")
        start_application()
        return
    
    # Run validation
    if not run_validation():
        print("\nValidation failed. Attempting to install dependencies...")
        
        if not os.path.exists('requirements.txt'):
            print("Error: requirements.txt not found!")
            print("Please ensure all project files are present.")
            return
        
        # Try to install dependencies
        if install_dependencies():
            print("\nRe-running validation...")
            if not run_validation():
                print("Validation still failing after dependency installation.")
                print("Please check the error messages above and fix manually.")
                return
        else:
            print("Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            return
    
    print("\n[OK] Validation passed! Starting application...")
    start_application()

if __name__ == "__main__":
    main()
