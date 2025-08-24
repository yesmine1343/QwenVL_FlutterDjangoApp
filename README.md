# Arabic OCR Script Setup

This script uses Hugging Face's Qwen2.5-VL model for Arabic text extraction from images.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   - Copy `token.env.example` to `token.env`
   - Edit `token.env` and replace `your_huggingface_token_here` with your actual Hugging Face token
   - Get your token from: https://huggingface.co/settings/tokens

3. **Run the Script**
   ```bash
   python independentforarabic.py
   ```

## Security Notes

- The `token.env` file is ignored by Git to keep your token secure
- Never commit your actual token to version control
- The `token.env.example` file serves as a template for others

## Files

- `independentforarabic.py` - Main script for Arabic OCR
- `token.env` - Environment file with your HF token (not tracked by Git)
- `token.env.example` - Template for environment setup
- `requirements.txt` - Python dependencies
- `file.gitignore` - Git ignore rules for sensitive files
