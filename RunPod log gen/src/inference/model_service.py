import json
import random
import base64
import requests

class MockVLMService:
    def __init__(self, model_name: str = "mock", device: str = "cpu", torch_dtype="auto"):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

    def run_inference(self, frame_paths, prompt_text: str) -> str:
        sample_outputs = [
            "EVENT: Normal office activity.\nOBJECT: People are seated at desks working on computers. No one is using a mobile phone.",
            "EVENT: Office interaction.\nOBJECT: Two people are standing near the aisle talking. One person is looking at a mobile phone."
        ]
        return random.choice(sample_outputs)


class LiquidVLMService:
    """
    Inference service for LiquidAI/LFM2.5-VL models.
    """
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype="bfloat16"):
        from PIL import Image
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.Image = Image
        self.torch = torch
        self.model_name = model_name
        self.device = device

        print(f"[VLM] Loading {model_name} ...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.image_token = "<image>"
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        tokenizer.image_start_token = "<|image_start|>"
        tokenizer.image_start_token_id = tokenizer.convert_tokens_to_ids("<|image_start|>")
        tokenizer.image_end_token = "<|image_end|>"
        tokenizer.image_end_token_id = tokenizer.convert_tokens_to_ids("<|image_end|>")
        tokenizer.image_thumbnail = "<|img_thumbnail|>"
        tokenizer.image_thumbnail_id = tokenizer.convert_tokens_to_ids("<|img_thumbnail|>")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )

        if device != "cuda":
            self.model.to(device)

        print(f"[VLM] {model_name} loaded on {device}")

    def run_inference(self, frame_paths, prompt_text: str) -> str:
        images = [self.Image.open(p).convert("RGB") for p in frame_paths]

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})

        conversation = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                min_p=0.15,
                repetition_penalty=1.05,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return output_text


class QwenVLMService:
    """
    Inference service for Qwen2.5-VL models.
    """
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype="bfloat16"):
        from PIL import Image
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        self.Image = Image
        self.torch = torch
        self.model_name = model_name
        self.device = device
        
        print(f"[VLM] Loading Qwen Model: {model_name} ...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if device != "cuda":
            self.model.to(device)
            
        print(f"[VLM] {model_name} loaded on {device}")
        
    def run_inference(self, frame_paths, prompt_text: str) -> str:
        from qwen_vl_utils import process_vision_info
        
        images = [self.Image.open(p).convert("RGB") for p in frame_paths]
        
        # Build Qwen-specific content array
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content}]
        
        # Qwen workflow
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                repetition_penalty=1.1,
                temperature=0.2,
                do_sample=True
            )
            
        # Extract output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()


class API_VLMService:
    """
    Inference service for an OpenAI-compatible API (e.g., vLLM).
    Supports multi-image requests natively.
    """
    def __init__(self, model_name: str, api_url: str = "http://localhost:8000/v1/chat/completions"):
        self.model_name = model_name
        self.api_url = api_url
        print(f"[VLM] Initialized API client for model: {model_name} at {api_url}")

    def run_inference(self, frame_paths, prompt_text: str) -> str:
        content = [{"type": "text", "text": prompt_text}]
        
        for path in frame_paths:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # We assume JPEG for simplicity, or it infers from extension
                img_url = f"data:image/jpeg;base64,{encoded_string}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 256,
            "temperature": 0.2
        }
        
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[VLM API ERROR] {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"[VLM API RESPONSE] {response.text}")
            return "EVENT: Error\nOBJECT: VLM API failed to respond."


def build_model_service(cfg_model: dict):
    mode = cfg_model.get("mode", "mock").lower()
    model_name = cfg_model.get("model_name", "mock")

    if mode == "mock":
        return MockVLMService(
            model_name=model_name,
            device=cfg_model.get("device", "cpu"),
            torch_dtype=cfg_model.get("torch_dtype", "auto")
        )
        
    if mode == "api":
        return API_VLMService(
            model_name=model_name,
            api_url=cfg_model.get("api_url", "http://localhost:8000/v1/chat/completions")
        )

    if mode == "real":
        if "qwen" in model_name.lower():
            return QwenVLMService(
                model_name=model_name,
                device=cfg_model.get("device", "cuda"),
                torch_dtype=cfg_model.get("torch_dtype", "bfloat16"),
            )
        else:
            return LiquidVLMService(
                model_name=model_name,
                device=cfg_model.get("device", "cuda"),
                torch_dtype=cfg_model.get("torch_dtype", "bfloat16"),
            )

    raise ValueError(f"Unsupported model mode: {mode}")