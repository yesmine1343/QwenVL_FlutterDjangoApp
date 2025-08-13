from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import logging
import tempfile
import os
import json

# Import your OCR functions - make sure these are properly implemented
try:
    from test_ocr import load_qwen_model, extract_text_qwen
    OCR_AVAILABLE = True
except ImportError as e:
    logging.error(f"OCR module import failed: {e}")
    OCR_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_model_cache = None

def get_cached_model():
    """Load and cache the Qwen model to avoid repeated loading."""
    global _model_cache
    if _model_cache is None and OCR_AVAILABLE:
        try:
            _model_cache = load_qwen_model()
            if _model_cache[0] is None:  # processor is None
                _model_cache = None
                logger.error("Failed to load Qwen model")
        except Exception as e:
            logger.error(f"Error loading Qwen model: {e}")
            _model_cache = None
    return _model_cache

@method_decorator(csrf_exempt, name='dispatch')
class QwenOCRView(View):
    def get(self, request):
        """GET endpoint to check service status"""
        status = "ready" if OCR_AVAILABLE and get_cached_model() else "unavailable"
        
        return JsonResponse({
            'message': 'Qwen OCR endpoint status check',
            'supported_languages': ['arabic', 'french', 'english', 'default'],
            'supported_formats': ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'],
            'max_file_size': '10MB',
            'status': status,
            'ocr_available': OCR_AVAILABLE
        })

    def post(self, request):
        """POST endpoint to process OCR requests"""
        try:
            # ENHANCED DEBUG LOGGING
            logger.info("=" * 50)
            logger.info("NEW OCR REQUEST RECEIVED")
            logger.info("=" * 50)
            
            # Log all request details
            logger.info(f"Content-Type: {request.content_type}")
            logger.info(f"Content-Length: {request.META.get('CONTENT_LENGTH', 'Not set')}")
            logger.info(f"HTTP_CONTENT_TYPE: {request.META.get('HTTP_CONTENT_TYPE', 'Not set')}")
            logger.info(f"Request method: {request.method}")
            
            # Log FILES dictionary
            logger.info(f"request.FILES keys: {list(request.FILES.keys())}")
            logger.info(f"request.FILES contents: {dict(request.FILES)}")
            
            # Log POST dictionary
            logger.info(f"request.POST keys: {list(request.POST.keys())}")
            logger.info(f"request.POST contents: {dict(request.POST)}")
            
            # Log request size from META instead of body (to avoid stream consumption)
            content_length = request.META.get('CONTENT_LENGTH')
            if content_length:
                logger.info(f"Request content length: {content_length} bytes")
            else:
                logger.info("No content length in request headers")
            
            # Check each file in FILES
            for key, file_obj in request.FILES.items():
                logger.info(f"File key '{key}': name='{file_obj.name}', size={file_obj.size}, content_type='{file_obj.content_type}'")
            
            # Check if OCR is available
            if not OCR_AVAILABLE:
                return JsonResponse({
                    'error': 'OCR service is not available. Please check server configuration.',
                    'status': 'error'
                }, status=503)

            # Enhanced file validation with more debugging
            if 'file' not in request.FILES:
                # Try alternative field names that might be used
                possible_keys = ['File', 'image', 'upload', 'document']
                found_file = None
                
                for key in possible_keys:
                    if key in request.FILES:
                        logger.info(f"Found file in alternative key: '{key}'")
                        found_file = request.FILES[key]
                        break
                
                if found_file:
                    file = found_file
                    logger.info("Using alternative file field")
                else:
                    return JsonResponse({
                        'error': 'No file uploaded. Please include a file in the "file" field.',
                        'status': 'error',
                        'debug_info': {
                            'received_files': list(request.FILES.keys()),
                            'received_post': list(request.POST.keys()),
                            'content_type': request.content_type,
                            'content_length': request.META.get('CONTENT_LENGTH'),
                            'expected_field': 'file',
                            'suggestion': 'Make sure Postman form-data field name is exactly "file"'
                        }
                    }, status=400)
            else:
                file = request.FILES['file']
                logger.info("Found file in expected 'file' field")
            
            # Enhanced file validation
            logger.info(f"File object type: {type(file)}")
            logger.info(f"File name: {file.name}")
            logger.info(f"File size: {file.size}")
            logger.info(f"File content type: {file.content_type}")
            
            # Validate file exists and has content
            if not file:
                return JsonResponse({
                    'error': 'File object is None',
                    'status': 'error'
                }, status=400)
                
            if file.size == 0:
                return JsonResponse({
                    'error': 'Uploaded file is empty',
                    'status': 'error',
                    'debug_info': {
                        'file_name': file.name,
                        'file_size': file.size
                    }
                }, status=400)
            
            # Validate file type
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff']
            file_content_type = file.content_type or 'unknown'
            
            # Also check file extension as fallback
            file_extension = os.path.splitext(file.name)[1].lower() if file.name else ''
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            logger.info(f"File extension: {file_extension}")
            logger.info(f"Content type validation: {file_content_type} in {allowed_types}")
            logger.info(f"Extension validation: {file_extension} in {allowed_extensions}")
            
            if (file_content_type not in allowed_types and 
                file_extension not in allowed_extensions):
                return JsonResponse({
                    'error': f'Invalid file type. Allowed types: {", ".join(allowed_types)}',
                    'received_type': file_content_type,
                    'received_extension': file_extension,
                    'status': 'error'
                }, status=400)
            
            # Validate file size (max 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if file.size > max_size:
                return JsonResponse({
                    'error': f'File too large. Maximum size is {max_size // (1024*1024)}MB',
                    'received_size_mb': round(file.size / (1024*1024), 2),
                    'status': 'error'
                }, status=400)
            
            # Get and validate language
            language_word = request.POST.get('language', 'default').lower().strip()
            valid_languages = ['arabic', 'french', 'english', 'default']
            if language_word not in valid_languages:
                return JsonResponse({
                    'error': f'Invalid language. Valid options: {", ".join(valid_languages)}',
                    'received_language': language_word,
                    'status': 'error'
                }, status=400)
            
            # Map language to prompt index
            language_prompt_map = {
                'arabic': 1,
                'french': 2,
                'english': 2,
                'default': 3
            }
            prompt_idx = language_prompt_map.get(language_word, 3)
            
            logger.info(f"Processing OCR request: language={language_word}, "
                       f"file_size={file.size}, content_type={file_content_type}, "
                       f"filename={file.name}")
            
            # Load model (using cache)
            model_data = get_cached_model()
            if not model_data:
                logger.error("Failed to load Qwen model")
                return JsonResponse({
                    'error': 'OCR model failed to load. Please try again later.',
                    'status': 'error'
                }, status=500)
            
            processor, model, model_id = model_data
            
            # Process the image
            try:
                # Reset file pointer to beginning
                file.seek(0)
                logger.info("Starting OCR processing...")
                
                # Process using your OCR function
                text, confidence, details = extract_text_qwen(
                    file, processor, model, model_id, 
                    language=language_word, 
                    prompt_idx=prompt_idx
                )
                
                logger.info("OCR processing completed")
                
            except Exception as ocr_error:
                logger.error(f"OCR processing error: {str(ocr_error)}", exc_info=True)
                return JsonResponse({
                    'error': f'OCR processing failed: {str(ocr_error)}',
                    'status': 'error'
                }, status=500)
            
            # Check if extraction was successful
            if isinstance(text, str) and text.startswith("Error"):
                logger.error(f"OCR extraction failed: {text}")
                return JsonResponse({
                    'error': text,
                    'status': 'error'
                }, status=500)
            
            # Ensure text is string and confidence is numeric
            text = str(text) if text is not None else ""
            confidence = float(confidence) if confidence is not None else 0.0
            
            # Prepare response
            response_data = {
                'text': text,
                'confidence': round(confidence, 3),
                'language': language_word,
                'model_used': model_id,
                'status': 'success',
                'file_info': {
                    'filename': file.name,
                    'size_bytes': file.size,
                    'content_type': file_content_type
                },
                'text_analysis': {
                    'text_length': len(text),
                    'has_arabic': any('\u0600' <= c <= '\u06FF' for c in text),
                    'has_french_accents': any(c in 'éèàçùâêîôûëïÿñÉÈÀÇÙÂÊÎÔÛËÏŸÑ' for c in text),
                    'word_count': len(text.split()) if text.strip() else 0
                }
            }
            
            # Add detailed metrics if available
            if details and isinstance(details, dict):
                response_data['details'] = {
                    'quality_metrics': details.get('text_quality_metrics', {}),
                    'consistency_score': details.get('consistency_score', 0),
                    'generated_variants': details.get('generated_variants', 1),
                    'individual_confidences': details.get('individual_confidences', []),
                    'best_prompt_variant': details.get('best_prompt_variant', 0)
                }
            
            logger.info(f"OCR processing completed successfully: confidence={confidence:.3f}, "
                       f"text_length={len(text)}")
            return JsonResponse(response_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return JsonResponse({
                'error': 'Invalid JSON in request',
                'status': 'error'
            }, status=400)
            
        except Exception as e:
            logger.error(f"Unexpected error in OCR processing: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'An unexpected error occurred during processing. Please try again.',
                'status': 'error',
                'debug_message': str(e) if logger.level <= logging.DEBUG else None
            }, status=500)

    def options(self, request):
        """Handle preflight CORS requests"""
        response = JsonResponse({'status': 'ok'})
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
        return response