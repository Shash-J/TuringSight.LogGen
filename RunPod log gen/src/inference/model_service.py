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
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype="auto"):
        from PIL import Image
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.Image = Image
        self.torch = torch
        self.model_name = model_name
        self.device = device

        if torch_dtype == "auto":
            dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )

        if device != "cuda":
            self.model.to(device)

    def run_inference(self, frame_paths, prompt_text: str) -> str:
        images = [self.Image.open(p).convert("RGB") for p in frame_paths]

        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256
        )

        output_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]

        return output_text.strip()


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
            torch_dtype=cfg_model.get("torch_dtype", "auto")
        )

    raise ValueError(f"Unsupported model mode: {mode}")