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
                token=None
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
                token=None
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