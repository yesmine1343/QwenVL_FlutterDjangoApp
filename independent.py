import os
import base64
from PIL import Image
import mimetypes
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or "YOUR_API_KEY_HERE")

# Choose model
model = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-2.5-flash" if available

def ocr_with_gemini(uploaded_file):
    # Read file bytes directly from Django file object
    uploaded_file.seek(0)  # Reset file pointer
    img_bytes = uploaded_file.read()
    
    # Detect mime type
    mime_type = uploaded_file.content_type

    if not mime_type or mime_type == "application/octet-stream":
        # Try to detect from file extension
        if hasattr(uploaded_file, 'name') and uploaded_file.name:
            if uploaded_file.name.lower().endswith('.png'):
                mime_type = "image/png"
            elif uploaded_file.name.lower().endswith('.gif'):
                mime_type = "image/gif"
            elif uploaded_file.name.lower().endswith('.webp'):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default fallback
        else:
            mime_type = "image/jpeg"
    
    # Build prompt in the correct schema
    prompt = [
        {
            "role": "user",
            "parts": [
                {"text": "Extract all text from this image. Write only the text."},
                {"inline_data": {"mime_type": mime_type, "data": img_bytes}}
            ]
        }
    ]
    
    # Generate response
    response = model.generate_content(prompt)
    return response.text

