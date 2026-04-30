import json
import random


class MockVLMService:
    def __init__(self, model_name: str = "mock", device: str = "cpu", torch_dtype="auto"):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

    def run_inference(self, frame_paths, prompt_text: str) -> str:
        sample_outputs = [
            {
                "event": "Normal office activity is visible with people seated at desks and occasional movement in the workspace.",
                "object": "Office workspace with desks, chairs, and visible employees."
            },
            {
                "event": "A few people appear to be working at their desks while one person is standing near the aisle.",
                "object": "Office area showing employees, workstations, and open walking space."
            },
            {
                "event": "Light interaction is visible in the office while other employees continue desk work.",
                "object": "Office scene with seated staff and a visible interaction near the work area."
            }
        ]
        return json.dumps(random.choice(sample_outputs))


class RealVLMService:
    """
    Inference service for LiquidAI/LFM2.5-VL-450M.

    Uses AutoModelForImageTextToText + AutoProcessor with the
    processor.apply_chat_template() workflow as per the official
    model card: https://huggingface.co/LiquidAI/LFM2.5-VL-450M
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
        """
        Run VLM inference on one or more frames.

        For multi-frame tasks (summaries), each frame is added as a
        separate image in the conversation content list.
        """
        images = [self.Image.open(p).convert("RGB") for p in frame_paths]

        # Build conversation content: one image entry per frame + prompt text
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})

        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # LFM2.5 pattern: apply_chat_template returns tokenized tensors directly
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

        # Slice off the input prompt tokens to get only the generated response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return output_text


def build_model_service(cfg_model: dict):
    mode = cfg_model.get("mode", "mock").lower()

    if mode == "mock":
        return MockVLMService(
            model_name=cfg_model.get("model_name", "mock"),
            device=cfg_model.get("device", "cpu"),
            torch_dtype=cfg_model.get("torch_dtype", "auto")
        )

    if mode == "real":
        return RealVLMService(
            model_name=cfg_model["model_name"],
            device=cfg_model.get("device", "cuda"),
            torch_dtype=cfg_model.get("torch_dtype", "bfloat16"),
        )

    raise ValueError(f"Unsupported model mode: {mode}")