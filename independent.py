import os
import cv2
import numpy as np
import json
from datetime import datetime
from PIL import Image
import torch
import psutil
from transformers.utils import logging
from transformers.file_utils import TRANSFORMERS_CACHE
import torch.nn.functional as F
from collections import Counter
from bidi.algorithm import get_display
import re
import torch
import arabic_reshaper
import tempfile
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Transformers cache directory:", TRANSFORMERS_CACHE)

def check_system_requirements():
    """
    Check if system meets requirements
    """
    print("🔍 Checking system requirements...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        try:
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU Memory: {props.total_memory / 1e9:.1f} GB")
        except:
            print("  GPU Memory: Unable to determine")
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")
    
    # Check memory
    try:
        memory = psutil.virtual_memory()
        print(f"✓ System RAM: {memory.total / 1e9:.1f} GB ({memory.percent}% used)")
        if memory.available < 4e9:  # Less than 4GB available
            print("⚠️  Low available RAM - consider closing other applications")
    except Exception as e:
        print(f"ℹ️  Could not check memory: {e}")
    
    print()

def preprocess_image_for_memory(image_path, max_size=1024):
    """
    Preprocess image to reduce memory usage while maintaining readability
    """
    try:
        # Open and convert image
        img = Image.open(image_path).convert("RGB")
        
        # Get original dimensions
        width, height = img.size
        print(f"Original image size: {width}x{height}")
        
        # Calculate new size maintaining aspect ratio
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Resize with high-quality resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized to: {new_width}x{new_height}")
        
        # Enhance contrast for better OCR
        import PIL.ImageEnhance
        enhancer = PIL.ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Slight contrast boost
        
        return img
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def calculate_sequence_probability(logits, input_ids, start_idx=0):
    """
    Calculate the probability of the generated sequence from logits
    """
    try:
        if logits is None or len(logits) == 0:
            return 0.75  # Default high confidence if no logits available
            
        total_log_prob = 0.0
        valid_tokens = 0
        
        for i, logit in enumerate(logits):
            if start_idx + i + 1 < len(input_ids):
                # Get the token that was actually generated
                target_token = input_ids[start_idx + i + 1]
                
                # Convert logits to probabilities
                probs = F.softmax(torch.tensor(logit), dim=-1)
                
                # Get probability of the actual token
                token_prob = probs[target_token].item()
                
                if token_prob > 0:
                    total_log_prob += np.log(token_prob)
                    valid_tokens += 1
        
        if valid_tokens == 0:
            return 0.75
            
        # Convert back to probability (geometric mean)
        avg_log_prob = total_log_prob / valid_tokens
        raw_prob = np.exp(avg_log_prob)
        
        # Apply confidence boost for reasonable probabilities
        if raw_prob < 0.1:
            boosted_conf = 0.5 + (np.log10(raw_prob * 10 + 1) / 2)
            return min(0.9, max(0.5, boosted_conf))
        else:
            return min(0.95, max(0.6, raw_prob))
        
    except Exception as e:
        print(f"Error calculating sequence probability: {e}")
        return 0.75

def calculate_text_quality_metrics(text):
    """
    Calculate various quality metrics for the extracted text
    """
    if not text or len(text.strip()) == 0:
        return {
            'length_score': 0.0,
            'character_ratio': 0.0,
            'word_ratio': 0.0,
            'repetition_penalty': 0.0,
            'overall_quality': 0.0
        }
    
    text = text.strip()
    
    # Length score (reasonable text length)
    if len(text) >= 20:
        length_score = 1.0
    else:
        length_score = min(len(text) / 20.0, 1.0)
    
    # Character quality (ratio of alphanumeric + punctuation to total)
    good_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:-\'\"éèàçù')
    character_ratio = good_chars / len(text) if len(text) > 0 else 0.0
    
    # Word quality
    words = text.split()
    if len(words) == 0:
        word_ratio = 0.0
    else:
        reasonable_words = sum(1 for word in words if len(word) >= 1)
        word_ratio = reasonable_words / len(words)
    
    # Repetition penalty
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    if total_words > 0:
        repetition_penalty = min(unique_words / total_words, 0.95)
    else:
        repetition_penalty = 0.0
    
    # French text bonus
    french_bonus = 1.0
    if any(c in text for c in 'éèàçùâêîôûëïÿñ'):
        french_bonus = 1.1
    
    # Overall quality score
    base_quality = (length_score * 0.15 + 
                   character_ratio * 0.35 + 
                   word_ratio * 0.35 + 
                   repetition_penalty * 0.15)
    
    overall_quality = min(base_quality * french_bonus, 1.0)
    
    return {
        'length_score': length_score,
        'character_ratio': character_ratio,
        'word_ratio': word_ratio,
        'repetition_penalty': repetition_penalty,
        'french_bonus': french_bonus,
        'overall_quality': overall_quality
    }

def load_qwen_model():
    """
    Load Qwen2-VL model with memory optimization
    """
    print("🔄 Loading Qwen2-VL model with memory optimization...")
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        model_id = "Qwen/Qwen2-VL-2B-Instruct"  # Start with smaller model
        
        # Load processor first
        print("  Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True,
            resume_download=True
        )
        
        # Load model with aggressive memory optimization
        print("  Loading model with memory optimization...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            resume_download=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            # Memory optimization parameters
            max_memory={0: "1GB"} if not torch.cuda.is_available() else None,
        )
        
        # Set model to eval mode to save memory
        model.eval()
        
        print(f"✓ Successfully loaded: {model_id}")
        return processor, model, model_id
        
    except Exception as e:
        print(f"✗ Failed to load Qwen model: {e}")
        
        # Try TrOCR fallback
        print("🔄 Trying TrOCR fallback...")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            
            print("✓ TrOCR fallback loaded successfully!")
            return processor, model, "microsoft/trocr-base-printed"
            
        except Exception as e:
            print(f"✗ TrOCR fallback also failed: {e}")
            return None, None, None

def reshape_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def contains_arabic(text):
    return any('\u0600' <= c <= '\u06FF' for c in text)

def extract_text_qwen_optimized(image_file, processor, model, model_id, language="default"):
    """
    Memory-optimized version of text extraction
    """
    try:
        # Load and preprocess image
        if isinstance(image_file, str):
            # Handle file path
            img = preprocess_image_for_memory(image_file)
        else:
            # Handle file-like object
            img = Image.open(image_file).convert("RGB")
            # Resize if too large
            width, height = img.size
            if max(width, height) > 1024:
                if width > height:
                    new_width = 1024
                    new_height = int(height * (1024 / width))
                else:
                    new_height = 1024
                    new_width = int(width * (1024 / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        if img is None:
            return "Error: Could not preprocess image", 0.0, {}
        
        # Check if using TrOCR fallback
        if "trocr" in model_id.lower():
            return extract_text_trocr(img, processor, model)
        
        # Memory-optimized prompts (shorter to reduce memory usage)
        if language == "arabic":
            prompt_text = "اقرأ النص في الصورة"
        elif language == "french":
            prompt_text = "Extraire avec précision tout le texte contenu dans l’image.Restituer uniquement le texte sans ajout ni interprétation."
        else:
            prompt_text = "Read all text in this image exactly"
        
        messages = {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text}
            ]
        }
        
        # Process with memory management
        text_prompt = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        
        # Clear any previous GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        inputs = processor(
            text=[text_prompt], 
            images=[img], 
            return_tensors="pt",
            padding=True
        )
        
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate with conservative settings to avoid memory issues
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,  # Reduced from 1024
                do_sample=False,
                temperature=1.0,
                pad_token_id=processor.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                # Remove output_attentions to save memory
            )
        
        # Decode the response
        generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
        text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Calculate confidence
        quality_metrics = calculate_text_quality_metrics(text)
        confidence = quality_metrics['overall_quality']
        
        # Sequence probability if available
        if hasattr(outputs, 'scores') and outputs.scores:
            try:
                seq_prob = calculate_sequence_probability(
                    [score[0].cpu().numpy() for score in outputs.scores],
                    outputs.sequences[0].cpu().numpy(),
                    len(inputs["input_ids"][0])
                )
                confidence = (confidence * 0.6 + seq_prob * 0.4)
            except:
                pass
        
        detailed_metrics = {
            'text_quality_metrics': quality_metrics,
            'model_type': 'Qwen2-VL-Optimized'
        }
        
        # Apply Arabic reshaping if needed
        if contains_arabic(text):
            text = reshape_text(text)
        
        # Clear memory
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return text, confidence, detailed_metrics
        
    except Exception as e:
        # Clear memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return f"Error processing image: {str(e)}", 0.0, {}

def extract_text_trocr(img, processor, model):
    """
    Fallback OCR using TrOCR with confidence estimation
    """
    try:
        pixel_values = processor(img, return_tensors="pt").pixel_values
        
        # Generate with scores for confidence calculation
        with torch.no_grad():
            outputs = model.generate(
                pixel_values, 
                max_length=500,  # Reduced from 1000
                output_scores=True, 
                return_dict_in_generate=True,
                do_sample=False
            )
        
        generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        
        # Calculate confidence based on text quality for TrOCR
        quality_metrics = calculate_text_quality_metrics(generated_text)
        confidence = quality_metrics['overall_quality'] * 0.85
        
        # Sequence probability if available
        if hasattr(outputs, 'scores') and outputs.scores:
            seq_prob = calculate_sequence_probability(
                [score[0].cpu().numpy() for score in outputs.scores],
                outputs.sequences[0].cpu().numpy()
            )
            confidence = (confidence + seq_prob) / 2
        
        detailed_metrics = {
            'text_quality_metrics': quality_metrics,
            'model_type': 'TrOCR'
        }
        
        return generated_text, confidence, detailed_metrics
        
    except Exception as e:
        return f"TrOCR Error: {str(e)}", 0.0, {}

def normalize_path(path):
    """
    Normalize file path for cross-platform compatibility
    """
    # Remove quotes and normalize path
    path = path.strip().strip('"').strip("'")
    return os.path.normpath(path)

def validate_image(image_path):
    """
    Validate if image can be processed
    """
    try:
        if not os.path.exists(image_path):
            return False, f"File does not exist: {image_path}"
        
        with Image.open(image_path) as img:
            img.verify()
        
        # Reopen for size check
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 50 or height < 50:
                return False, "Image too small (minimum 50x50)"
            if width * height > 10000 * 10000:  # Very large images
                return False, "Image too large, will be automatically resized"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

def test_single_image_optimized(image_path):
    """
    Process a single image with memory optimization and detailed analysis
    """
    # Normalize the path
    image_path = normalize_path(image_path)
    
    if not os.path.exists(image_path):
        print(f"✗ Image {image_path} does not exist!")
        print(f"Checked path: {os.path.abspath(image_path)}")
        return

    print("🔄 Initializing model...")
    try:
        processor, model, model_id = load_qwen_model()
        if processor is None:
            print("✗ Failed to load any model!")
            return
        print(f"✓ Model loaded successfully: {model_id}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Validate image
    is_valid, validation_msg = validate_image(image_path)
    if not is_valid and "will be automatically resized" not in validation_msg:
        print(f"✗ {validation_msg}")
        return
    
    print(f"🖼️  Processing: {os.path.basename(image_path)}")
    
    # Try to detect language from filename or ask user
    language = "default"
    if "ar" in image_path.lower() or "arabic" in image_path.lower():
        language = "arabic"
    elif "fr" in image_path.lower() or "french" in image_path.lower():
        language = "french"
    
    text, confidence, detailed_metrics = extract_text_qwen_optimized(
        image_path, processor, model, model_id, language=language
    )

    print("\n" + "=" * 60)
    print("📊 DETAILED RESULTS")
    print("=" * 60)
    
    if text.startswith("Error"):
        print(f"✗ {text}")
    else:
        print(f"✓ Overall Confidence: {confidence:.3f}")
        print(f"✓ Model used: {model_id}")
        
        if 'text_quality_metrics' in detailed_metrics:
            tq = detailed_metrics['text_quality_metrics']
            print(f"📝 Text Quality Breakdown:")
            print(f"   Character Quality: {tq['character_ratio']:.3f}")
            print(f"   Word Quality: {tq['word_ratio']:.3f}")
            print(f"   Uniqueness: {tq['repetition_penalty']:.3f}")
            print(f"   Length Score: {tq['length_score']:.3f}")
        
        print("\n📝 Extracted Text:")
        print("-" * 40)
        print(text)
        print("-" * 40)

def run_ocr_optimized(image_file, language="default"):
    """
    Memory-optimized version for API usage
    """
    try:
        processor, model, model_id = load_qwen_model()
        if processor is None:
            return {"error": "Model failed to load"}
    except Exception as e:
        return {"error": f"Model loading failed: {e}"}

    try:
        text, confidence, details = extract_text_qwen_optimized(
            image_file, processor, model, model_id, language=language
        )
        return {
            "text": text,
            "confidence": confidence,
            "details": details
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🚀 Enhanced Qwen-VL OCR Script with Memory Optimization")
    print("=" * 60)

    check_system_requirements()

    image_path = input("📁 Enter full path to image: ").strip()
    if image_path:
        test_single_image_optimized(image_path)


if __name__ == "__main__":
    main()