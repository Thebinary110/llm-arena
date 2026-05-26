"""Modal serverless deployment for Qwen2.5-0.5B-Instruct.

Deploy with:
    modal deploy deployment/modal_app.py

The web endpoint URL becomes the MODAL_ENDPOINT env var consumed by OSSAssistant.
"""

from __future__ import annotations

import modal

app = modal.App("qwen-assistant")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45.0",
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "fastapi",
        "uvicorn",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
class QwenModel:
    """Holds a loaded Qwen2.5-0.5B-Instruct model for serverless inference.

    The model is loaded once when the container starts (@modal.enter) and
    reused across concurrent requests until the container is recycled.
    """

    @modal.enter()
    def load_model(self) -> None:
        """Load tokeniser and model into GPU memory on container start."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    @modal.method()
    def generate(self, messages: list[dict], max_tokens: int = 512) -> dict:
        """Run chat-template inference and return content + token count.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            max_tokens: Maximum new tokens to generate.

        Returns:
            {"content": str, "tokens_used": int}
        """
        import torch

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = generated_ids[0][input_len:]
        content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_used = len(generated_tokens)

        return {"content": content, "tokens_used": tokens_used}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def chat_endpoint(request: dict) -> dict:
    """HTTP POST endpoint consumed by OSSAssistant when USE_MODAL=True.

    Expected request body:
        {"messages": [...], "max_tokens": int (optional)}

    Returns:
        {"content": str, "tokens_used": int}
    """
    model = QwenModel()
    return model.generate.remote(
        request["messages"],
        request.get("max_tokens", 512),
    )
