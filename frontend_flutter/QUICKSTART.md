# Quick Start Guide

Get your Qwen 2.5 Handwritten Image Reader up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- Flutter SDK
- At least 8GB RAM (for model loading)

## Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Run the setup script**:
   ```bash
   python setup.py
   ```

2. **Start the backend**:
   ```bash
   cd backend
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python manage.py runserver
   ```

3. **Start the Flutter app** (in a new terminal):
   ```bash
   flutter run
   ```

### Option 2: Manual Setup

#### Backend Setup

1. **Navigate to backend**:
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

4. **Start server**:
   ```bash
   python manage.py runserver
   ```

#### Frontend Setup

1. **Install Flutter dependencies**:
   ```bash
   flutter pub get
   ```

2. **Run the app**:
   ```bash
   flutter run
   ```

## First Use

1. **Select Language**: Choose the language of your text (English, French, Arabic, or Other)
2. **Pick Image**: Select an image from your device
3. **Wait for Processing**: The app will upload and process your image
4. **View Results**: See the extracted text with confidence score

## Troubleshooting

### Backend Issues

- **Port already in use**: Change port with `python manage.py runserver 8001`
- **Model loading fails**: Ensure you have enough RAM and internet connection
- **CORS errors**: Check that Django server is running on `http://127.0.0.1:8000`

### Frontend Issues

- **Connection errors**: Verify backend is running and accessible
- **File picker issues**: Check app permissions on mobile devices
- **Build errors**: Run `flutter clean && flutter pub get`

## API Testing

Test the API directly:

```bash
curl -X POST http://127.0.0.1:8000/api/qwen-ocr/ \
  -F "file=@your_image.jpg" \
  -F "language=english"
```

## Performance Tips

- Use images under 5MB for faster processing
- Close other applications to free up RAM
- Use GPU if available for faster inference

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure sufficient system resources
4. Check network connectivity for model downloads 