#!/usr/bin/env python3
"""
Setup script for Qwen 2.5 Handwritten Image Reader
"""

import os
import sys
import subprocess
import platform

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def setup_backend():
    """Setup Django backend"""
    print("\n🔧 Setting up Django Backend...")
    
    if not os.path.exists("backend"):
        print("❌ Backend directory not found")
        return False
    
    os.chdir("backend")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        if not run_command("python -m venv venv"):
            return False
    
    # Activate virtual environment and install dependencies
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print("Installing Python dependencies...")
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        return False
    
    print("Running Django migrations...")
    if not run_command(f"{pip_cmd} install django-cors-headers"):
        return False
    
    if not run_command(f"{pip_cmd} install djangorestframework"):
        return False
    
    print("✅ Backend setup completed")
    os.chdir("..")
    return True

def setup_frontend():
    """Setup Flutter frontend"""
    print("\n📱 Setting up Flutter Frontend...")
    
    # Check if Flutter is installed
    if not run_command("flutter --version"):
        print("❌ Flutter is not installed. Please install Flutter first:")
        print("   https://docs.flutter.dev/get-started/install")
        return False
    
    print("Installing Flutter dependencies...")
    if not run_command("flutter pub get"):
        return False
    
    print("✅ Frontend setup completed")
    return True

def main():
    """Main setup function"""
    print("🚀 Qwen 2.5 Handwritten Image Reader Setup")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    if not setup_backend():
        print("❌ Backend setup failed")
        sys.exit(1)
    
    if not setup_frontend():
        print("❌ Frontend setup failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the Django backend:")
    print("   cd backend")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("   python manage.py runserver")
    print("\n2. Start the Flutter app:")
    print("   flutter run")
    print("\n3. Open the app and start using OCR!")

if __name__ == "__main__":
    main() 