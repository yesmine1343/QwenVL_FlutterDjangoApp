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
        if memory.available < 8e9:  # Less than 8GB available
            print("⚠️  Low available RAM - consider closing other applications")
    except Exception as e:
        print(f"ℹ️  Could not check memory: {e}")
    
    print()

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
        # Transform very low probabilities to more reasonable confidence scores
        if raw_prob < 0.1:
            # Boost very low probabilities using log scaling
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
    
    # Length score (reasonable text length) - more generous scoring
    if len(text) >= 20:
        length_score = 1.0  # Full score for 20+ characters
    else:
        length_score = min(len(text) / 20.0, 1.0)
    
    # Character quality (ratio of alphanumeric + punctuation to total)
    good_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:-\'\"éèàçù')
    character_ratio = good_chars / len(text) if len(text) > 0 else 0.0
    
    # Word quality (reasonable words vs total tokens) - more lenient
    words = text.split()
    if len(words) == 0:
        word_ratio = 0.0
    else:
        reasonable_words = sum(1 for word in words if len(word) >= 1)  # Changed from > 1 to >= 1
        word_ratio = reasonable_words / len(words)
    
    # Repetition penalty (detect repetitive patterns) - less harsh
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    if total_words > 0:
        repetition_penalty = min(unique_words / total_words, 0.95)  # Cap at 0.95
    else:
        repetition_penalty = 0.0
    
    # French text bonus - check for French characteristics
    french_bonus = 1.0
    if any(c in text for c in 'éèàçùâêîôûëïÿñ'):
        french_bonus = 1.1  # 10% bonus for French accents
    
    # Overall quality score with adjusted weights and French bonus
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
        
        # Calculate entropy of attention distribution (lower entropy = more focused = higher confidence)
        attention_flat = attention_weights.flatten()
        attention_flat = attention_flat[attention_flat > 0]  # Remove zeros
        
        if len(attention_flat) == 0:
            return 0.5
        
        # Normalize
        attention_probs = attention_flat / np.sum(attention_flat)
        
        # Calculate entropy
        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-10))
        
        # Convert entropy to confidence (lower entropy = higher confidence)
        max_entropy = np.log(len(attention_probs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        return max(0.0, min(1.0, confidence))
        
    except Exception as e:
        print(f"Error calculating attention confidence: {e}")
        return 0.5

_QWEN_CACHE = {}

def load_qwen_model():
    """
    Load Qwen2-VL model with proper error handling and fallbacks
    """
    global _QWEN_CACHE
    if _QWEN_CACHE.get("ok"):
        return _QWEN_CACHE["processor"], _QWEN_CACHE["model"], _QWEN_CACHE["model_id"]

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    tried = [
        ("Qwen/Qwen2-VL-2B-Instruct", torch.float16),
        ("Qwen/Qwen2-VL-7B-Instruct", torch.float16),
    ]
    for mid, dtype in tried:
        try:
            processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True, resume_download=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                mid,
                torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            _QWEN_CACHE = {"ok": True, "processor": processor, "model": model, "model_id": mid}
            return processor, model, mid

        except ImportError as e:
            print(f"  ✗ Import error: {e}")
            print("  Installing required packages...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "--upgrade", "transformers", "torch", "torchvision"])
                print("  Packages updated, please restart the script")
                return None, None, None
            except:
                print("  Please manually install: pip install --upgrade transformers torch torchvision")
                continue

        except Exception as e:
            print(f"  ✗ Failed to load {mid}: {str(e)[:100]}...")
            continue

    # If all Qwen models fail, try TrOCR as fallback
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
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def contains_arabic(text):
    return any('\u0600' <= c <= '\u06FF' for c in text)

def extract_text_qwen(image_file, processor, model, model_id, language="default", prompt_idx=None):
    """
    Use Qwen-VL or TrOCR to extract text from an image file-like object with improved Arabic support.
    image_file: file-like object (e.g., Django InMemoryUploadedFile, Flask FileStorage, or Python file object)
    prompt_idx: 1-based index (1=arabic, 2=french/english, 3=default). If None, use language logic.
    
    Why check for None? If prompt_idx is provided, it allows the API/frontend to explicitly control which prompt variant to use. If not provided (None), the function falls back to language-based selection for backward compatibility and flexibility.
    """
    try:
        # Load and preprocess image from file-like object
        try:
            img = Image.open(image_file).convert("RGB")
        except Exception as e:
            return f"Error: Unable to open image file: {e}", 0.0, {}
        
        # Check if using TrOCR fallback
        if "trocr" in model_id.lower():
            return extract_text_trocr(img, processor, model)
        
        model_device = next(model.parameters()).device
        # Enhanced prompt with multiple attempts for different language priorities
        prompt_variants = [
            # Variant 1: Language-neutral approach
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Extract all text from this image exactly as written. Maintain original formatting, spacing, and preserve all characters including Arabic, French, English, numbers, and special characters. Output only the text content without any explanations."}
                ]
            },
            # Variant 2: Arabic-focused
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "اقرأ النص في هذه الصورة بدقة. احترم اتجاه الكتابة العربية من اليمين إلى اليسار وشكل الحروف. أعطِ النص فقط. لا تعكس ترتيب الحروف أو الكلمات."}
                ]
            },
            # Variant 3: French-focused  
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Lisez précisément tout le texte manuscrit ou imprimé dans cette image, en respectant les accents français et la ponctuation. Donnez uniquement le contenu textuel exact."}
                ]
            }
        ]
        # Select prompt by prompt_idx if provided, else by language
        
    
        if language == "arabic":
            selected_prompt = prompt_variants[1]
        elif language == "french" or language == "english":
            selected_prompt = prompt_variants[2]
        else:
            selected_prompt = prompt_variants[0]

    # Try each prompt variant and collect results
        all_results = []
        messages = selected_prompt
        i = prompt_variants.index(messages)
        try:
            # Prepare inputs for this variant
            text_prompt = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            variant_inputs = processor(
                text=[text_prompt], 
                images=[img], 
                return_tensors="pt",
                padding=True
            )
            # Move to device if needed
            variant_inputs = {k: v.to(model_device) for k, v in variant_inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **variant_inputs, 
                    max_new_tokens=1024,
                    do_sample=False,  # Use greedy decoding for consistency
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True
                )
            # Decode the response
            generated_ids = outputs.sequences[0][len(variant_inputs["input_ids"][0]):]
            text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            # Calculate quality score for this variant
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
        
        # Select the best result based on quality score
        best_result = max(all_results, key=lambda x: x['quality_score'])
        # Use the best result for the rest of the processing
        messages = prompt_variants[best_result['variant_id']]
        text_prompt = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        inputs = best_result['inputs']
        # Now generate additional samples from the best prompt for confidence estimation
        confidence_scores = []
        generated_texts = [best_result['text']]  # Start with the best variant result
        # Generate 2 more samples using the best prompt with slight variations
        for sample in range(2):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=True,  # Use sampling for variation
                    temperature=0.3,   # Low temperature for consistency
                    top_p=0.8,        # Focused sampling
                    pad_token_id=processor.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True
                )
            # Decode the response
            generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
            text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            generated_texts.append(text)
            # Calculate confidence metrics
            confidence_metrics = {}
            # 1. Sequence probability confidence
            if hasattr(outputs, 'scores') and outputs.scores:
                seq_prob = calculate_sequence_probability(
                    [score[0].cpu().numpy() for score in outputs.scores],
                    outputs.sequences[0].cpu().numpy(),
                    len(inputs["input_ids"][0])
                )
                confidence_metrics['sequence_probability'] = seq_prob
            # 2. Text quality confidence
            quality_metrics = calculate_text_quality_metrics(text)
            confidence_metrics['text_quality'] = quality_metrics['overall_quality']
            # 3. Attention-based confidence (if available)
            if hasattr(outputs, 'attentions') and outputs.attentions:
                try:
                    # Use the last layer's attention
                    last_attention = outputs.attentions[-1][-1]  # Last layer, last head
                    attention_conf = calculate_attention_based_confidence(last_attention)
                    confidence_metrics['attention_confidence'] = attention_conf
                except:
                    confidence_metrics['attention_confidence'] = 0.5
            # Combine confidence scores with better weighting
            combined_confidence = (
                confidence_metrics.get('sequence_probability', 0.75) * 0.4 +  # Reduced weight
                confidence_metrics.get('text_quality', 0.75) * 0.5 +          # Increased weight  
                confidence_metrics.get('attention_confidence', 0.75) * 0.1     # Minimal weight
            )
            confidence_scores.append(combined_confidence)
        # Now calculate confidence for all generated texts
        for i, text in enumerate(generated_texts):
            confidence_metrics = {}
            # Use the outputs from the corresponding generation
            if i == 0:  # First text is from the best variant
                current_outputs = best_result['outputs']
                current_inputs = best_result['inputs']
            else:  # Later texts from additional sampling
                current_outputs = outputs  # Uses the last outputs from the loop
                current_inputs = inputs
            # 1. Sequence probability confidence
            if hasattr(current_outputs, 'scores') and current_outputs.scores:
                seq_prob = calculate_sequence_probability(
                    [score[0].cpu().numpy() for score in current_outputs.scores],
                    current_outputs.sequences[0].cpu().numpy(),
                    len(current_inputs["input_ids"][0])
                )
                confidence_metrics['sequence_probability'] = seq_prob
            # 2. Text quality confidence
            quality_metrics = calculate_text_quality_metrics(text)
            confidence_metrics['text_quality'] = quality_metrics['overall_quality']
            # 3. Attention-based confidence (if available)
            if hasattr(current_outputs, 'attentions') and current_outputs.attentions:
                try:
                    # Use the last layer's attention
                    last_attention = current_outputs.attentions[-1][-1]  # Last layer, last head
                    attention_conf = calculate_attention_based_confidence(last_attention)
                    confidence_metrics['attention_confidence'] = attention_conf
                except:
                    confidence_metrics['attention_confidence'] = 0.5
            # Combine confidence scores with better weighting
            combined_confidence = (
                confidence_metrics.get('sequence_probability', 0.75) * 0.4 +
                confidence_metrics.get('text_quality', 0.75) * 0.5 +
                confidence_metrics.get('attention_confidence', 0.75) * 0.1
            )
            confidence_scores.append(combined_confidence)
        # Select the best result based on confidence (might be different from variant selection)
        best_idx = np.argmax(confidence_scores)
        best_text = generated_texts[best_idx]
        best_confidence = confidence_scores[best_idx]
        # Calculate consistency confidence (how similar are the generated texts)
        consistency_score = calculate_consistency_score(generated_texts)
        # Final confidence combines individual confidence with consistency
        final_confidence = (best_confidence * 0.8 + consistency_score * 0.2)
        # Apply minimum confidence threshold for good quality text
        quality_metrics = calculate_text_quality_metrics(best_text)
        if quality_metrics['overall_quality'] > 0.9 and consistency_score > 0.9:
            final_confidence = max(final_confidence, 0.85)  # Boost high quality results
        # Additional metrics for debugging/analysis
        detailed_metrics = {
            'individual_confidences': confidence_scores,
            'consistency_score': consistency_score,
            'generated_variants': len(set(generated_texts)),
            'text_quality_metrics': quality_metrics,
            'best_prompt_variant': best_result['variant_id'],
            'prompt_variant_scores': [r['quality_score'] for r in all_results]
        }
        # Apply Arabic reshaping if needed
        if contains_arabic(best_text):
            best_text = reshape_text(best_text)
        return best_text, final_confidence, detailed_metrics
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0, {}
    


def calculate_consistency_score(texts):
    """
    Calculate how consistent the generated texts are
    """
    if len(texts) <= 1:
        return 1.0
    
    # Remove duplicates
    unique_texts = list(set(texts))
    
    if len(unique_texts) == 1:
        return 1.0  # Perfect consistency
    
    # Calculate average similarity between texts
    from difflib import SequenceMatcher
    
    total_similarity = 0
    comparisons = 0
    
    for i in range(len(unique_texts)):
        for j in range(i + 1, len(unique_texts)):
            similarity = SequenceMatcher(None, unique_texts[i], unique_texts[j]).ratio()
            total_similarity += similarity
            comparisons += 1
    
    return total_similarity / comparisons if comparisons > 0 else 0.0
    import math

def preprocess_image_pil(pil_img, target_max_side=1600, binarize=True, deskew=True):
    """Return a cleaned PIL.Image for better handwriting OCR."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # fast denoise + local contrast
    gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # optional adaptive binarization (helps pale ink)
    if binarize:
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        # remove isolated dots / salt
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        work = thr
    else:
        work = gray

    # optional deskew (cheap)
    if deskew:
        coords = np.column_stack(np.where(work < 255))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            (h, w) = work.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            work = cv2.warpAffine(work, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # upscale a bit helps thin strokes
    h, w = work.shape[:2]
    scale = min(target_max_side / max(h, w), 2.0)
    if scale < 1.0 or scale > 1.0:
        work = cv2.resize(work, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # back to RGB PIL
    if binarize:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
    else:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(work)

def merge_tile_texts(tile_texts, min_line_len=2):
    """Simple overlap merge: dedupe by line content while keeping order."""
    seen = set()
    lines_out = []
    for _, txt in sorted(tile_texts, key=lambda t: (t[0][1], t[0][0])):  # sort by y then x
        for ln in [l.strip() for l in txt.splitlines() if len(l.strip()) >= min_line_len]:
            key = ln
            if key not in seen:
                seen.add(key)
                lines_out.append(ln)
    return "\n".join(lines_out).strip()


def extract_text_qwen_v2(pil_img, processor, model, model_id, language="default",
                         use_tiling=True, max_new_tokens=320):
    # 1) preprocess
    clean = preprocess_image_pil(pil_img, target_max_side=1600, binarize=True, deskew=True)

    prompts = {
        "default": "Extract all text exactly as written. Keep line breaks. Output only text.",
        "arabic": "اقرأ النص كما هو تمامًا مع الحفاظ على الأسطر والاتجاه من اليمين إلى اليسار. أعد النص فقط.",
        "french": "Extrait tout le texte tel qu'écrit en gardant les retours à la ligne. Donne uniquement le texte.",
        "english": "Extract all text exactly as written, preserving line breaks. Output text only.",
    }
    ptxt = prompts.get(language, prompts["default"])

    def run_one(pil_tile):
        msg = [{"role":"user","content":[{"type":"image","image":pil_tile},{"type":"text","text":ptxt}]}]
        text_prompt = processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[pil_tile], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )
        gen_ids = out.sequences[0][len(inputs["input_ids"][0]):]
        txt = processor.decode(gen_ids, skip_special_tokens=True).strip()
        return txt

    if not use_tiling:
        txt = run_one(clean)
        if contains_arabic(txt): txt = reshape_text(txt)
        qm = calculate_text_quality_metrics(txt)
        return txt, max(0.6, qm["overall_quality"]), {"tiled": False, "text_quality_metrics": qm}

    # tiling path
    tiles = list(tile_image(clean, tile=960, overlap=96))
    tile_texts = []
    for (tile_pil, box) in tiles:
        try:
            t = run_one(tile_pil)
        except Exception as e:
            t = ""
        tile_texts.append((box, t))

    merged = merge_tile_texts(tile_texts)
    if contains_arabic(merged): merged = reshape_text(merged)
    qm = calculate_text_quality_metrics(merged)
    # light confidence: quality + simple consistency among top tiles
    conf = max(0.6, min(0.95, qm["overall_quality"]))
    return merged, conf, {"tiled": True, "n_tiles": len(tiles), "text_quality_metrics": qm}


def validate_image(image_path):
    """
    Validate if image can be processed
    """
    try:
        with Image.open(image_path) as img:
            # Check if image is not corrupted
            img.verify()
        
        # Reopen for size check (verify() closes the image)
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 50 or height < 50:
                return False, "Image too small (minimum 50x50)"
            if width * height > 4096 * 4096:  # Very large images
                return False, "Image too large, consider resizing"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

def save_results_to_file(results, output_file):
    """
    Save results to JSON file with proper error handling
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")


def test_single_image(image_path):
    """
    Process a single image with detailed confidence analysis
    """
    # Validate image first
    is_valid, validation_msg = validate_image(image_path)
    if not is_valid:
        print(f"✗ {validation_msg}")
        return

    print(f"🖼️  Processing: {os.path.basename(image_path)}")

    # Open and process image
    with Image.open(image_path).convert("RGB") as pil_img:
        text, confidence, detailed_metrics = extract_text_qwen_v2(
            pil_img, processor, model, model_id,
            language="arabic" if "arab" in image_path.lower() else "default",
            use_tiling=True, max_new_tokens=320
        )

    print("\n" + "=" * 60)
    print("📊 DETAILED RESULTS")
    print("=" * 60)
    if text.startswith("Error"):
        print(f"✗ {text}")
    else:
        print(f"✓ Overall Confidence: {confidence:.3f}")
        print(f"✓ Model used: {model_id}")
        
        if 'individual_confidences' in detailed_metrics:
            print(f"🔍 Individual Samples: {[f'{c:.3f}' for c in detailed_metrics['individual_confidences']]}")
        
        if 'consistency_score' in detailed_metrics:
            print(f"🔄 Consistency Score: {detailed_metrics['consistency_score']:.3f}")
        
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

def run_ocr(image_file, language="default"):
    """
    image_file: Django InMemoryUploadedFile or similar
    language: string from frontend like 'arabic', 'french', 'english'
    """

    # Save the file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        for chunk in image_file.chunks():
            temp.write(chunk)
        temp_path = temp.name

    try:
        # Load model
        processor, model, model_id = load_qwen_model()
        if processor is None:
            return {"error": "Model failed to load"}

        # Validate image
        is_valid, validation_msg = validate_image(temp_path)
        if not is_valid:
            return {"error": validation_msg}

        # Run OCR (choose ONE function)
        text, confidence, details = extract_text_qwen(
            temp_path, processor, model, model_id, language=language
        )

        return {
            "text": text,
            "confidence": confidence,
            "details": details
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        os.remove(temp_path)  # Always cleanup


def main():
    print("🚀 Enhanced Qwen-VL OCR Script with Dynamic Confidence")
    print("=" * 60)
    # Check system requirements
    check_system_requirements()
    image_path = input("📁 Enter full path to image: ").strip().strip('"')
    test_single_image(image_path)


if __name__ == "__main__":
    main()