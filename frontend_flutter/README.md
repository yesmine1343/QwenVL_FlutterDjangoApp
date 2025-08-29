# Qwen 2.5 Handwritten Image Reader

A Flutter application that uses Qwen 2.5 Vision Language Model for OCR (Optical Character Recognition) with support for multiple languages including Arabic, French, and English.

## Features

- **Multi-language OCR**: Supports Arabic, French, English, and other languages
- **File Upload**: Pick images from device gallery or camera
- **Real-time Processing**: Upload and process images through Django API
- **Confidence Scoring**: Shows confidence level for extracted text
- **Modern UI**: Clean, responsive Flutter interface
- **Error Handling**: Comprehensive error handling and user feedback

## Architecture

- **Frontend**: Flutter app with file picker and HTTP client
- **Backend**: Django REST API with Qwen 2.5 VL model
- **OCR Engine**: Qwen 2.5 Vision Language Model for text extraction

## Setup Instructions

### Backend Setup (Django)

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Django migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Start Django server**:
   ```bash
   python manage.py runserver
   ```

The Django server will run on `http://127.0.0.1:8000`

### Frontend Setup (Flutter)

1. **Navigate to Flutter project root**:
   ```bash
   cd flutter_qwen_app
   ```

2. **Install Flutter dependencies**:
   ```bash
   flutter pub get
   ```

3. **Run Flutter app**:
   ```bash
   flutter run
   ```

## API Endpoints

### POST `/api/qwen-ocr/`

Extract text from an uploaded image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: Image file (JPEG, PNG, BMP, TIFF)
  - `language`: Language code (`arabic`, `french`, `english`, `default`)

**Response**:
```json
{
  "text": "Extracted text content",
  "confidence": 0.85,
  "language": "english",
  "model_used": "Qwen/Qwen2-VL-2B-Instruct",
  "status": "success",
  "details": {
    "text_length": 45,
    "has_arabic": false,
    "has_french_accents": false,
    "quality_metrics": {...},
    "consistency_score": 0.92,
    "generated_variants": 3
  }
}
```

## Usage

1. **Select Language**: Choose the language of the text in your image
2. **Pick Image**: Select an image file from your device
3. **Process**: The app automatically uploads and processes the image
4. **View Results**: See the extracted text with confidence score
5. **Copy Text**: Use the copy button to copy extracted text

## Supported Languages

- **English**: Optimized for English text extraction
- **Français**: Optimized for French text with accents
- **العربية**: Optimized for Arabic text with right-to-left support
- **Other Language**: General purpose for other languages

## File Requirements

- **Supported Formats**: JPEG, JPG, PNG, BMP, TIFF
- **Maximum Size**: 10MB
- **Minimum Resolution**: 50x50 pixels

## Error Handling

The application provides comprehensive error handling for:
- File upload issues
- Network connectivity problems
- Model loading failures
- Invalid file types or sizes
- Processing errors

## Development Notes

### Backend
- Uses Django REST Framework for API
- CORS enabled for Flutter requests
- Comprehensive logging and error handling
- Model caching for performance

### Frontend
- Material Design 3 UI
- Responsive layout
- Real-time status updates
- Error feedback with snackbars

## Troubleshooting

### Common Issues

1. **Model Loading Fails**:
   - Ensure sufficient RAM (8GB+ recommended)
   - Check internet connection for model download
   - Try restarting the Django server

2. **CORS Errors**:
   - Verify Django server is running on correct port
   - Check CORS settings in `settings.py`
   - Ensure Flutter app is using correct API URL

3. **File Upload Issues**:
   - Check file size (max 10MB)
   - Verify file format is supported
   - Ensure proper file permissions

### Performance Tips

- Use smaller images for faster processing
- Close other applications to free up RAM
- Use GPU if available for faster model inference

## License

This project is for educational and research purposes.
