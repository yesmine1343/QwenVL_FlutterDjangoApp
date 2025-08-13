#!/usr/bin/env python3
"""
Enhanced Test script for Django OCR API
Run this script to test your OCR endpoint with a sample image
"""

import requests
import json
import os
from pathlib import Path
import io
from PIL import Image

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
OCR_ENDPOINT = "/api/ocrapi/"  # Change this to match your URL pattern

def create_test_image():
    """Create a simple test image with text for testing"""
    print("📝 Creating test image with text...")
    
    # Create a simple image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        # Try to load a better font (works on most systems)
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)  # Linux
            except:
                font = ImageFont.load_default()
    
    # Add text to image
    text_lines = [
        "Hello World!",
        "This is a test image",
        "with multiple lines of text",
        "for OCR testing"
    ]
    
    y_position = 20
    for line in text_lines:
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 30
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def create_french_test_image():
    """Create a test image with French text"""
    print("📝 Creating French test image...")
    
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (500, 250), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    french_lines = [
        "Bonjour le monde!",
        "Voici un texte français",
        "avec des accents: é, è, à, ç",
        "pour tester l'OCR",
        "Merci beaucoup!"
    ]
    
    y_position = 20
    for line in french_lines:
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 35
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_get_endpoint():
    """Test the GET endpoint to check service status"""
    print("🔍 Testing GET endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}{OCR_ENDPOINT}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ GET request failed: {e}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON response: {response.text}")
        return False

def test_post_endpoint_no_file():
    """Test POST endpoint without file to see error handling"""
    print("\n🔍 Testing POST endpoint without file...")
    try:
        response = requests.post(f"{API_BASE_URL}{OCR_ENDPOINT}", 
                               data={'language': 'english'})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ POST request failed: {e}")
        return False

def test_post_endpoint_with_sample_image():
    """Test POST endpoint with a sample image"""
    print("\n🔍 Testing POST endpoint with sample image...")
    
    try:
        # Create test image in memory
        img_bytes = create_test_image()
        
        files = {'file': ('test_image.png', img_bytes, 'image/png')}
        data = {'language': 'english'}
        
        response = requests.post(f"{API_BASE_URL}{OCR_ENDPOINT}", 
                               files=files, data=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"Response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
            
            # Show extracted text clearly if successful
            if response.status_code == 200 and 'text' in response_json:
                print(f"\n📝 Extracted Text: '{response_json['text']}'")
                print(f"🎯 Confidence: {response_json.get('confidence', 'N/A')}")
                
        except json.JSONDecodeError:
            print(f"Raw Response: {response.text}")
            
        return response.status_code == 200
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out - OCR processing may be taking too long")
        return False
    except Exception as e:
        print(f"❌ POST request with file failed: {e}")
        return False

def test_french_image():
    """Test with French text image"""
    print("\n🔍 Testing with French text image...")
    
    try:
        img_bytes = create_french_test_image()
        
        files = {'file': ('french_test.png', img_bytes, 'image/png')}
        data = {'language': 'french'}
        
        response = requests.post(f"{API_BASE_URL}{OCR_ENDPOINT}", 
                               files=files, data=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"Response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200 and 'text' in response_json:
                print(f"\n📝 Extracted French Text: '{response_json['text']}'")
                print(f"🎯 Confidence: {response_json.get('confidence', 'N/A')}")
                
        except json.JSONDecodeError:
            print(f"Raw Response: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ French text test failed: {e}")
        return False

def test_with_real_image_file():
    """Test with a real image file if available"""
    print("\n🔍 Testing with real image file (if available)...")
    
    # Common image file locations to try
    possible_files = [
        "test_image.jpg", "test_image.png", "sample.jpg", "sample.png",
        "image.jpg", "image.png", "test.jpg", "test.png"
    ]
    
    image_file = None
    for filename in possible_files:
        if os.path.exists(filename):
            image_file = filename
            break
    
    if not image_file:
        print("⏭️ No test image file found, skipping...")
        return True  # Not a failure
    
    try:
        print(f"📁 Using image file: {image_file}")
        
        with open(image_file, 'rb') as f:
            files = {'file': (image_file, f, 'image/jpeg')}
            data = {'language': 'default'}
            
            response = requests.post(f"{API_BASE_URL}{OCR_ENDPOINT}", 
                                   files=files, data=data, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"Response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200 and 'text' in response_json:
                print(f"\n📝 Extracted Text: '{response_json['text']}'")
                print(f"🎯 Confidence: {response_json.get('confidence', 'N/A')}")
                
        except json.JSONDecodeError:
            print(f"Raw Response: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        return False

def test_with_curl_examples():
    """Print curl commands for manual testing"""
    print(f"\n📋 Manual testing with curl:")
    print(f"GET test:")
    print(f'curl -X GET "{API_BASE_URL}{OCR_ENDPOINT}"')
    
    print(f"\nPOST test (replace 'your_image.jpg' with actual image path):")
    print(f'curl -X POST "{API_BASE_URL}{OCR_ENDPOINT}" \\')
    print(f'  -F "file=@"C:\testAI\fasterDAN\Datasets\raw2\Screenshot 2025-07-10 165750.png" \\')
    print(f'  -F "language=english"')

def test_model_loading():
    """Test if the model can be loaded by making a simple GET request"""
    print("\n🔍 Testing model loading status...")
    try:
        response = requests.get(f"{API_BASE_URL}{OCR_ENDPOINT}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service Status: {data.get('status', 'unknown')}")
            print(f"✅ OCR Available: {data.get('ocr_available', 'unknown')}")
            return data.get('status') == 'ready'
        else:
            print(f"❌ Service returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not check service status: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Enhanced OCR API Tests")
    print(f"Testing endpoint: {API_BASE_URL}{OCR_ENDPOINT}")
    print("=" * 60)
    
    # Check if PIL is available
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL/Pillow is available for test image creation")
    except ImportError:
        print("❌ PIL/Pillow not available. Install with: pip install Pillow")
        return
    
    # Test service status
    model_ready = test_model_loading()
    
    # Test GET endpoint
    get_success = test_get_endpoint()
    
    # Test POST without file
    post_no_file_success = test_post_endpoint_no_file()
    
    # Test POST with generated image
    post_with_file_success = test_post_endpoint_with_sample_image()
    
    # Test French text
    french_success = test_french_image()
    
    # Test with real image file
    real_file_success = test_with_real_image_file()
    
    # Show curl examples
    test_with_curl_examples()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  🔧 Model Ready: {'PASS' if model_ready else 'FAIL'}")
    print(f"  ✅ GET endpoint: {'PASS' if get_success else 'FAIL'}")
    print(f"  ✅ POST no file: {'PASS' if post_no_file_success else 'FAIL'}")
    print(f"  ✅ POST generated image: {'PASS' if post_with_file_success else 'FAIL'}")
    print(f"  ✅ French text test: {'PASS' if french_success else 'FAIL'}")
    print(f"  ✅ Real file test: {'PASS' if real_file_success else 'SKIP'}")
    
    # Troubleshooting tips
    if not get_success or not model_ready:
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure Django server is running: python manage.py runserver 127.0.0.1:8000")
        print("  2. Check if the URL pattern in urls.py matches OCR_ENDPOINT")
        print("  3. Verify the view is properly imported in urls.py")
        print("  4. Check Django logs for model loading errors")
        print("  5. Ensure required packages are installed:")
        print("     pip install torch transformers pillow qwen-vl")
    
    if not post_with_file_success and get_success:
        print("\n🔧 OCR Processing Issues:")
        print("  1. Check if Qwen model downloaded correctly")
        print("  2. Verify CUDA/GPU setup if using GPU")
        print("  3. Monitor memory usage - OCR models require significant RAM")
        print("  4. Check Django logs for detailed error messages")

if __name__ == "__main__":
    main()