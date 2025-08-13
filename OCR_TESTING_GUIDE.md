# OCR API Testing Guide

This guide explains how to test the OCR API endpoint and use the enhanced Flutter app with detailed logging.

## 🚀 Enhanced Flutter App Features

### What's New:
1. **Detailed Console Logging**: All API requests and responses are logged to the debug console
2. **Raw JSON Viewer**: Click "View JSON" button to see the complete API response
3. **Enhanced Error Logging**: Better error messages with stack traces
4. **Success Logging**: Detailed success information including confidence and model used

### How to Use:

1. **Run the Flutter App**:
   ```bash
   cd flutter_qwen_app
   flutter run
   ```

2. **Select Image and Language**: Use the file picker to select an image and choose a language

3. **Process Image**: Click "Process Image" to send the request to the backend

4. **View Results**:
   - **In App**: The extracted text and confidence will be displayed in the UI
   - **In Console**: Detailed logs will appear in the debug console
   - **Raw JSON**: Click "View JSON" button to see the complete API response

### Console Output Examples:

**Successful OCR:**
```
=== OCR API RESPONSE ===
Response status: 200
Response headers: {content-type: application/json, ...}
Response data: {text: "Hello World", confidence: 0.95, ...}
========================

=== OCR SUCCESS ===
Extracted text: Hello World
Confidence: 0.95
Language: english
Model used: Qwen/Qwen2-VL-2B-Instruct
===================
```

**Error Response:**
```
=== OCR ERROR ===
Error message: File too large. Maximum size is 10MB
Status code: 400
Full response: {error: "File too large...", status: "error"}
================
```

## 🧪 Testing the OCR Endpoint

### Option 1: Using the Test Script

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r test_requirements.txt
   ```

2. **Start Django Server**:
   ```bash
   python manage.py runserver
   ```

3. **Run the Test Script**:
   ```bash
   python test_ocrapi_endpoint.py
   ```

4. **Add Test Images**: Place some image files (.jpg, .png, etc.) in the backend directory

### Option 2: Using curl

```bash
# Test endpoint health
curl -X GET http://127.0.0.1:8000/api/ocrapi/

# Test OCR with image
curl -X POST http://127.0.0.1:8000/api/ocrapi/ \
  -F "file=@your_image.jpg" \
  -F "language=english"
```

### Option 3: Using Postman

1. **URL**: `http://127.0.0.1:8000/api/ocrapi/`
2. **Method**: POST
3. **Body**: form-data
   - Key: `file` (Type: File) - Select your image
   - Key: `language` (Type: Text) - Enter: english, french, arabic, or default

## 📋 Test Script Features

The `test_ocrapi_endpoint.py` script provides:

1. **Health Check**: Tests if the endpoint is accessible
2. **Auto Image Detection**: Finds test images in the directory
3. **Multi-language Testing**: Tests all supported languages
4. **Error Case Testing**: Tests various error scenarios
5. **Detailed Output**: Shows request/response details

### Test Script Output Example:

```
🚀 OCR API Endpoint Test Script
==================================================
🔍 Testing endpoint health...
✅ GET request successful: 200
Response: {'message': 'Qwen OCR endpoint is ready...', 'status': 'ready'}

🔍 Looking for test images...
✅ Found 2 test images:
  - test_image.jpg
  - sample.png

🔍 Testing OCR with file: test_image.jpg
Language: english
📤 Sending POST request to: http://127.0.0.1:8000/api/ocrapi/
📁 File: test_image.jpg
🌐 Language: english
📥 Response status: 200

=== OCR RESPONSE ===
{
  "text": "Hello World",
  "confidence": 0.95,
  "language": "english",
  "model_used": "Qwen/Qwen2-VL-2B-Instruct",
  "status": "success"
}
===================
✅ OCR successful!
📝 Extracted text: Hello World
🎯 Confidence: 0.95
🌐 Language: english
🤖 Model: Qwen/Qwen2-VL-2B-Instruct
```

## 🔧 Troubleshooting

### Common Issues:

1. **Connection Error**: Make sure Django server is running on port 8000
2. **File Not Found**: Ensure test images are in the correct directory
3. **Model Loading Error**: Check if the Qwen model is properly installed
4. **Large File Error**: Images must be under 10MB

### Debug Steps:

1. **Check Django Server Logs**: Look for errors in the Django terminal
2. **Check Flutter Console**: View detailed logs in the Flutter debug console
3. **Test with curl**: Use curl to isolate if the issue is with the app or API
4. **Check File Permissions**: Ensure the script can read test images

## 📊 Expected Results

### Successful Response:
```json
{
  "text": "Extracted text from image",
  "confidence": 0.95,
  "language": "english",
  "model_used": "Qwen/Qwen2-VL-2B-Instruct",
  "status": "success"
}
```

### Error Response:
```json
{
  "error": "Error description",
  "status": "error"
}
```

## 🎯 Next Steps

1. **Test with Real Images**: Use actual images with text in different languages
2. **Monitor Performance**: Check processing time and memory usage
3. **Test Edge Cases**: Try very small images, images with no text, etc.
4. **Integration Testing**: Test the complete Flutter → Django → OCR flow 