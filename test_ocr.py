import os
import cv2
import numpy as np
import json
import tempfile  # 🔧 ADDED - was missing
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Transformers cache directory:", TRANSFORMERS_CACHE)

def check_system_requirements():
    print("🔍 Checking system requirements...")
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
        if memory.available < 8e9:  # Less than 8GB available
            print("⚠️  Low available RAM - consider closing other applications")
    except Exception as e:
        print(f"ℹ️  Could not check memory: {e}")
    
    print()

def calculate_sequence_probability(logits, input_ids, start_idx=0):
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
    if not text or len(text.strip()) == 0:
        return {
            'length_score': 0.0,
            'character_ratio': 0.0,
            'word_ratio': 0.0,
            'repetition_penalty': 0.0,
            'overall_quality': 0.0
        }
    
    text = text.strip()
    # Length score (reasonable text length) - more generous scoring
    if len(text) >= 20:
        length_score = 1.0
    else:
        length_score = min(len(text) / 20.0, 1.0)  
    good_chars = sum(1 for c in text if (
        c.isalnum() or c.isspace() or 
        c in '.,!?;:-\'\"éèàçù' or
        '\u0600' <= c <= '\u06FF' or  # Arabic
        '\u0750' <= c <= '\u077F' or  # Arabic Supplement  
        '\u08A0' <= c <= '\u08FF' or  # Arabic Extended-A
        '\uFB50' <= c <= '\uFDFF' or  # Arabic Presentation Forms-A
        '\uFE70' <= c <= '\uFEFF'     # Arabic Presentation Forms-B
    ))
    character_ratio = good_chars / len(text) if len(text) > 0 else 0.0
    
    # Word quality - more lenient for Arabic
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
    
    # Language bonuses
    arabic_bonus = 1.0
    french_bonus = 1.0
    
    # Arabic text bonus
    if contains_arabic(text):
        arabic_bonus = 1.15  # 15% bonus for Arabic text
        
    # French text bonus
    if any(c in text for c in 'éèàçùâêîôûëïÿñ'):
        french_bonus = 1.1  # 10% bonus for French accents
    
    # Overall quality score
    base_quality = (length_score * 0.15 + 
                   character_ratio * 0.35 + 
                   word_ratio * 0.35 + 
                   repetition_penalty * 0.15)
    
    overall_quality = min(base_quality * arabic_bonus * french_bonus, 1.0)
    
    return {
        'length_score': length_score,
        'character_ratio': character_ratio,
        'word_ratio': word_ratio,
        'repetition_penalty': repetition_penalty,
        'arabic_bonus': arabic_bonus,
        'french_bonus': french_bonus,
        'overall_quality': overall_quality
    }

def calculate_attention_based_confidence(attention_weights):
    """
    Calculate confidence based on attention weight distribution
    """
    try:
        if attention_weights is None:
            return 0.5
        
        # Convert to numpy if it's a tensor
        if hasattr(attention_weights, 'cpu'):
            attention_weights = attention_weights.cpu().numpy()
        
        # Calculate entropy of attention distribution
        attention_flat = attention_weights.flatten()
        attention_flat = attention_flat[attention_flat > 0]
        
        if len(attention_flat) == 0:
            return 0.5
        
        # Normalize
        attention_probs = attention_flat / np.sum(attention_flat)
        
        # Calculate entropy
        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-10))
        
        # Convert entropy to confidence
        max_entropy = np.log(len(attention_probs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        return max(0.0, min(1.0, confidence))
        
    except Exception as e:
        print(f"Error calculating attention confidence: {e}")
        return 0.5

def load_qwen_model():    
    # HTTP/API note: To reduce first-request latency and download size for
    # GET/POST on `api/ocrapi/`, we now try ONLY 1 model :
    # 2) Qwen/Qwen2-VL-2B-Instruct .
    models_to_try = [
       
        {
            "id": "Qwen/Qwen2-VL-2B-Instruct",
            "description": "Qwen2-VL 2B (Fallback - smallest if 2.5 cannot load)"
        }
    ]
    
    for model_info in models_to_try:
        model_id = model_info["id"]
        description = model_info["description"]
        
        print(f"🔄 Trying to load: {description}")
        print(f"🔄 Model ID: {model_id}")
        
        try:
            # Import for Qwen models
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # Load processor first
            print("  Loading processor...")
            processor = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True,
                resume_download=True,
                use_auth_token=False  # 🔧 Try without authentication
            )
            
            # Load model with optimizations
            print("  Loading model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
                output_attentions=True,
                use_safetensors=True,
                use_auth_token=False  # 🔧 Try without authentication
            )
            print(f"✓ Successfully loaded: {description}")
            print(f"✓ Model ID confirmed: {model_id}")
            return processor, model, model_id
        except ImportError as e:
            print(f"  ✗ Import error: {e}")
            print("  Try: pip install --upgrade transformers>=4.45.0 qwen-vl-utils")
            continue
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed to load {model_id}: {error_msg}")
            print(f"  Full error: {e}")
            continue
    
    # Fallback to TrOCR
    print("🔄 All Qwen models failed, trying TrOCR fallback...")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        
        print("✓ TrOCR fallback loaded successfully!")
        return processor, model, "microsoft/trocr-base-printed"
        
    except Exception as e:
        print(f"✗ Even TrOCR fallback failed: {e}")
        return None, None, None

def reshape_text(text):
    try:
        # First, reshape Arabic characters to their proper forms
        reshaped_text = arabic_reshaper.reshape(text)
        # Then apply bidirectional algorithm for proper RTL display
        display_text = get_display(reshaped_text)
        # Debug output to see what's happening
        print("reshaping test")
        print(f"🔄 Original: {repr(text[:30])}")
        print(f"🔄 Reshaped: {repr(reshaped_text[:30])}")
        print(f"🔄 Display:  {repr(display_text[:30])}")
        
        return display_text
    except Exception as e:
        print(f"Warning: Text reshaping failed: {e}")
        return text

def contains_arabic(text):
    if not text:
        return False
    arabic_ranges = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        (0x1EE00, 0x1EEFF) # Arabic Mathematical Alphabetic Symbols
    ]
    
    for char in text:
        char_code = ord(char)
        for start, end in arabic_ranges:
            if start <= char_code <= end:
                return True
    return False

# Just add this ONE line to your existing fix_arabic_ocr_issues function:

def fix_arabic_ocr_issues(text):
    if not contains_arabic(text):
        return text    
    # 🔧 ADD THIS LINE - Remove Japanese characters that leak into Arabic OCR
    import re
    text = re.sub(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', '', text)
    corrections = {
        # Reversed reading corrections
        'ﺃﺮﻗﺃ': 'أقرأ',
        'ﻥﺃ': 'أن',
        'ﺩﻮﺗ': 'تود',
        # ... rest of existing corrections
        'ﺢﻴﺤﺻ': 'صحيح',
        'ﻞﻜﺸﺑ': 'بشكل',
        'ﺺﻨﻟﺍ': 'النص',
        'ﺔﺑﺎﺘﻛ': 'كتابة',
        'ﺓﺩﺎﻋﺇ': 'إعادة',
        'ﻭﺃ': 'أو',
        'ﻼﻴﺼﻔﺗ': 'تفصيلا',
        'ﺮﺜﻛﺃ': 'أكثر',
        'ﻪﻣﺪﻘﺗ': 'تقدمه',
        'ﻢﻬﻓ': 'فهم',
        'ﻊﻴﻄﺘﺳﺃ': 'أستطيع'
    }
    
    fixed_text = text
    for wrong, correct in corrections.items():
        fixed_text = fixed_text.replace(wrong, correct)
    
    # Additional processing for mixed scripts
    if '申し' in fixed_text:  # Keep this existing check
        fixed_text = fixed_text.replace('申し', '')
    
    return fixed_text
def debug_arabic_detection(text):
    print(f"\n🔍 DEBUG: Arabic Detection Analysis")
    print(f"Raw text length: {len(text)}")
    print(f"First 50 chars: {repr(text[:50])}")
    arabic_found = False
    for i, char in enumerate(text[:20]):
        char_code = ord(char)
        is_arabic = any(start <= char_code <= end for start, end in [
            (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)
        ])
        if is_arabic:
            arabic_found = True
        print(f"Char {i:2d}: '{char}' -> U+{char_code:04X} -> Arabic: {is_arabic}")
    
    detection_result = contains_arabic(text)
    print(f"🎯 Arabic Detection Result: {detection_result}")
    print(f"🎯 Any Arabic Found in Sample: {arabic_found}")
    return detection_result

def extract_text_qwen(image_file, processor, model, model_id, language="default", prompt_idx=None):
    try:
        # Handle both file paths and file objects
        if isinstance(image_file, str):
            img = Image.open(image_file).convert("RGB")
        else:
            img = Image.open(image_file).convert("RGB")
        
        # Check if using TrOCR fallback
        if "trocr" in model_id.lower():
            return extract_text_trocr(img, processor, model)
        
        model_device = next(model.parameters()).device
        
        # Enhanced prompts for better Arabic handling
        prompt_variants = [
            # Variant 0: Language-neutral with RTL awareness
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Extract all text from this image exactly as written. Maintain original formatting, spacing, and preserve all characters including Arabic (right-to-left), French, English, numbers, and special characters. For Arabic text, preserve the correct reading direction. Output only the text content."}
                ]
            },
            # Variant 1: Enhanced Arabic-focused with better instructions
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "اقرأ النص العربي في هذه الصورة بدقة تامة. سواء كان النص مكتوباً باليد أو بالكمبيوتر، احترم اتجاه الكتابة من اليمين إلى اليسار والتشكيل الصحيح للحروف العربية. حافظ على ترتيب الكلمات والجمل كما هي في الأصل. تأكد من أن الحروف العربية متصلة بشكل صحيح. أعطِ النص فقط دون أي شرح أو ترجمة."}
                ]
            },
            # Variant : French-focused
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Lisez précisément tout le texte dans cette image, en respectant les accents français et la ponctuation. Donnez uniquement le contenu textuel exact."}
                ]
            }
        ]
        
        # 🔧 FIXED: Detect Arabic in image to select appropriate prompt
        # Convert image to text for language detection
        try:
            # Quick OCR to detect language in image
            quick_prompt = {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What language is the text in this image? Answer with only: arabic, english, french, or other."}
                ]
            }
            
            quick_text_prompt = processor.apply_chat_template([quick_prompt], tokenize=False, add_generation_prompt=True)
            quick_inputs = processor(
                text=[quick_text_prompt], 
                images=[img], 
                return_tensors="pt",
                padding=True
            )
            quick_inputs = {k: v.to(model_device) for k, v in quick_inputs.items()}
            
            with torch.no_grad():
                quick_outputs = model.generate(
                    **quick_inputs, 
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            quick_response = processor.decode(quick_outputs.sequences[0][len(quick_inputs["input_ids"][0]):], skip_special_tokens=True).strip().lower()
            print(f"🔍 Language detection response: '{quick_response}'")
            
            # Determine language from response
            detected_language = "default"
            if "arabic" in quick_response:
                detected_language = "arabic"
            elif "french" in quick_response:
                detected_language = "french"
            elif "english" in quick_response:
                detected_language = "english"
            
            print(f"🔍 Detected language: {detected_language}")
            
        except Exception as e:
            print(f"⚠️ Language detection failed: {e}")
            detected_language = language  # Fallback to provided language
        
        # Select prompt based on detected language or provided language
        if detected_language == "arabic" or language == "arabic":
            selected_prompt = prompt_variants[1]
            print("🔄 Using Arabic-focused prompt")
        elif detected_language == "french" or language == "french" or detected_language == "english" or language == "english":
            selected_prompt = prompt_variants[2]
            print("🔄 Using French-focused prompt")
        else:
            selected_prompt = prompt_variants[0]
            print("🔄 Using general prompt")

        # Try the selected prompt
        all_results = []
        messages = selected_prompt
        i = prompt_variants.index(messages)
        
        try:
            # Prepare inputs
            text_prompt = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            variant_inputs = processor(
                text=[text_prompt], 
                images=[img], 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            variant_inputs = {k: v.to(model_device) for k, v in variant_inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **variant_inputs, 
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True
                )
            
            # Decode the response
            generated_ids = outputs.sequences[0][len(variant_inputs["input_ids"][0]):]
            text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Calculate quality metrics
            quality_metrics = calculate_text_quality_metrics(text)
            all_results.append({
                'text': text,
                'quality_score': quality_metrics['overall_quality'],
                'variant_id': i,
                'outputs': outputs,
                'inputs': variant_inputs,
                'quality_metrics': quality_metrics
            })
            
        except Exception as e:
            print(f"Variant {i} failed: {e}")
        
        if not all_results:
            return "Error: All prompt variants failed", 0.0, {}
        
        # Use the best result
        best_result = all_results[0]  # Only one result in this simplified version
        best_text = best_result['text']
        
        # Calculate confidence (simplified)
        confidence = best_result['quality_metrics']['overall_quality']
        
        # 🔧 ENHANCED: Debug Arabic detection
        print(f"\n🔍 Before Arabic processing: {repr(best_text[:100])}")
        debug_arabic_detection(best_text)
        
        # Apply Arabic reshaping if needed
        if contains_arabic(best_text):
            print("🔄 ✅ Arabic detected! Applying fixes...")
            try:
                original_text = best_text
                
                # First fix common OCR issues
                best_text = fix_arabic_ocr_issues(best_text)
                print(f"🔄 ✅ OCR fixes applied!")
                
                # Then apply reshaping
                best_text = reshape_text(best_text)
                print(f"🔄 ✅ Reshaping applied!")
                print(f"🔄 Before: {repr(original_text[:50])}")
                print(f"🔄 After:  {repr(best_text[:50])}")
            except Exception as e:
                print(f"⚠️ Arabic processing failed: {e}")
        else:
            print("❌ No Arabic detected - no reshaping applied")
        
        detailed_metrics = {
            'text_quality_metrics': best_result['quality_metrics'],
            'arabic_detected': contains_arabic(best_text),
            'model_version': '2.5' if '2.5' in model_id else '2.0'
        }
        
        return best_text, confidence, detailed_metrics
        
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0, {}

def calculate_consistency_score(texts):
    """Calculate consistency between generated texts"""
    if len(texts) <= 1:
        return 1.0
    
    unique_texts = list(set(texts))
    if len(unique_texts) == 1:
        return 1.0
    
    from difflib import SequenceMatcher
    total_similarity = 0
    comparisons = 0
    
    for i in range(len(unique_texts)):
        for j in range(i + 1, len(unique_texts)):
            similarity = SequenceMatcher(None, unique_texts[i], unique_texts[j]).ratio()
            total_similarity += similarity
            comparisons += 1
    
    return total_similarity / comparisons if comparisons > 0 else 0.0

def extract_text_trocr(img, processor, model):
    """Fallback OCR using TrOCR"""
    try:
        pixel_values = processor(img, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values, 
                max_length=1000, 
                output_scores=True, 
                return_dict_in_generate=True,
                do_sample=False
            )
        
        generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        
        quality_metrics = calculate_text_quality_metrics(generated_text)
        confidence = quality_metrics['overall_quality'] * 0.85
        
        detailed_metrics = {
            'text_quality_metrics': quality_metrics,
            'model_type': 'TrOCR'
        }
        
        return generated_text, confidence, detailed_metrics
        
    except Exception as e:
        return f"TrOCR Error: {str(e)}", 0.0, {}

def validate_image(image_file):
    """
    🔧 FIXED: Validate image file - handles both paths and file objects
    """
    try:
        if isinstance(image_file, str):
            # File path
            with Image.open(image_file) as img:
                img.verify()
            with Image.open(image_file) as img:
                width, height = img.size
                if width < 50 or height < 50:
                    return False, "Image too small (minimum 50x50)"
                if width * height > 4096 * 4096:
                    return False, "Image too large, consider resizing"
        else:
            # File object
            current_pos = image_file.tell()
            image_file.seek(0)
            
            with Image.open(image_file) as img:
                img.verify()
            
            image_file.seek(0)
            with Image.open(image_file) as img:
                width, height = img.size
                if width < 50 or height < 50:
                    return False, "Image too small (minimum 50x50)"
                if width * height > 4096 * 4096:
                    return False, "Image too large, consider resizing"
            
            image_file.seek(current_pos)
        
        return True, "Valid"
        
    except Exception as e:
        if not isinstance(image_file, str):
            try:
                image_file.seek(current_pos)
            except:
                pass
        return False, f"Invalid image: {str(e)}"

def save_results_to_file(results, output_file):
    """Save results to JSON file"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

def test_single_image(image_file):
    """Process a single image file with detailed analysis"""
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
    is_valid, validation_msg = validate_image(image_file)
    if not is_valid:
        print(f"✗ {validation_msg}")
        return

    # Get filename for display
    if isinstance(image_file, str):
        filename = os.path.basename(image_file)
    else:
        filename = getattr(image_file, 'filename', 'uploaded_image') or 'uploaded_image'
    
    print(f"🖼️  Processing: {filename}")
    
    text, confidence, detailed_metrics = extract_text_qwen(image_file, processor, model, model_id)

    print("\n" + "=" * 60)
    print("📊 DETAILED RESULTS")
    print("=" * 60)
    if text.startswith("Error"):
        print(f"✗ {text}")
    else:
        print(f"✓ Overall Confidence: {confidence:.3f}")
        print(f"✓ Model used: {model_id}")
        print(f"✓ Arabic detected: {detailed_metrics.get('arabic_detected', False)}")

        if 'text_quality_metrics' in detailed_metrics:
            tq = detailed_metrics['text_quality_metrics']
            print(f"📝 Text Quality Breakdown:")
            print(f"   Character Quality: {tq['character_ratio']:.3f}")
            print(f"   Word Quality: {tq['word_ratio']:.3f}")
            print(f"   Uniqueness: {tq['repetition_penalty']:.3f}")
            print(f"   Length Score: {tq['length_score']:.3f}")
            if 'arabic_bonus' in tq:
                print(f"   Arabic Bonus: {tq['arabic_bonus']:.3f}")

        print("\n📝 Extracted Text:")
        print("-" * 40)
        print(text)

def run_ocr(image_file, language="default"):
    """
    🔧 FIXED: Main OCR function for API integration
    """
    # Create temp file from uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        for chunk in image_file.chunks():
            temp.write(chunk)
        temp_path = temp.name

    try:
        processor, model, model_id = load_qwen_model()
        if processor is None:
            return {"error": "Model failed to load"}
        
        print(f"🤖 Loaded model: {model_id}")
        
    except Exception as e:
        return {"error": f"Model loading failed: {e}"}

    # 🔧 FIXED: Validate using temp_path (string) correctly
    is_valid, validation_msg = validate_image(temp_path)
    if not is_valid:
        return {"error": validation_msg}

    try:
        # Extract text using temp_path
        text, confidence, details = extract_text_qwen(temp_path, processor, model, model_id, language=language)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Enhanced response with proper Arabic detection
        response = {
            "text": text,
            "confidence": confidence,
            "language": language,
            "model_used": model_id,
            "status": "success",
            "text_analysis": {
                "text_length": len(text),
                "has_arabic": contains_arabic(text),  # 🔧 This should now work!
                "has_french_accents": any(c in text for c in 'éèàçùâêîôûëïÿñ'),
                "word_count": len(text.split()) if text else 0
            },
            "details": details
        }
        
        # Debug output
        print(f"🔍 Final Arabic detection: {contains_arabic(text)}")
        if contains_arabic(text):
            print(f"✅ Arabic confirmed in response!")
        else:
            print(f"❌ No Arabic in final response: {text[:50]}...")
            
        return response
        
    except Exception as e:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        return {"error": str(e)}

def main():
    print("🚀 FIXED Qwen-VL OCR Script with Working Arabic Detection")
    print("=" * 70)
    
    check_system_requirements()
    
    # Check if user wants to force download 2.5 model
    print("\n🔧 Model Download Options:")
    print("1. Use existing models (default)")
    print("2. Force download Qwen2.5-VL-3B-Instruct")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("\n🔄 Force downloading Qwen2.5-VL-3B-Instruct...")
        processor, model, model_id = force_download_model("Qwen/Qwen2.5-VL-3B-Instruct")
        if processor is None:
            print("✗ Failed to download 2.5 model. Continuing with existing models...")
        else:
            print("✓ 2.5 model downloaded successfully!")
            print("You can now restart the server to use the new model.")
            return
    
    while True:
        image_path = input("📁 Enter full path to image (or 'quit' to exit): ").strip().strip('"')
        
        if image_path.lower() == 'quit':
            print("👋 Goodbye!")
            break
            
        if not os.path.exists(image_path):
            print(f"✗ File not found: {image_path}")
            continue
            
        test_single_image(image_path)
        
        while True:
            another = input("\n🔄 Process another image? (y/n): ").strip().lower()
            if another in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' or 'n'")
        
        if another in ['n', 'no']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()