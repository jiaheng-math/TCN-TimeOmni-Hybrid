from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid_rul.llm_output import normalize_llm_response


def build_chat_prompt(question: str, system_prompt: str, assistant_prefix: str = "") -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


@dataclass
class GenerationConfig:
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    assistant_prefix: str = "<risk_summary>\n"
    retry_on_invalid: bool = True


class TimeOmniAdapter:
    def __init__(self, model_dir: str | None, generation_config: GenerationConfig) -> None:
        self.model_dir = model_dir
        self.generation_config = generation_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    @property
    def enabled(self) -> bool:
        return bool(self.model_dir)

    def load(self) -> None:
        if not self.enabled:
            return
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def _generate_once(
        self,
        question: str,
        system_prompt: str,
        assistant_prefix: str,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        prompt = build_chat_prompt(question, system_prompt, assistant_prefix=assistant_prefix)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        use_sampling = temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.max_new_tokens,
                do_sample=use_sampling,
                temperature=temperature if use_sampling else None,
                top_p=top_p if use_sampling else None,
                repetition_penalty=repetition_penalty,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return f"{assistant_prefix}{generated_text}".strip()

    def _should_retry(self, response: str) -> bool:
        audit = normalize_llm_response(response)
        return not audit["clean_format_ok"] and (
            audit["thought_block_count"] > 0 or "<thought>" in audit["raw_text"].lower()
        )

    def generate(self, question: str, system_prompt: str) -> str | None:
        if not self.enabled:
            return None
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        response = self._generate_once(
            question=question,
            system_prompt=system_prompt,
            assistant_prefix=self.generation_config.assistant_prefix,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            repetition_penalty=self.generation_config.repetition_penalty,
        )
        if not self.generation_config.retry_on_invalid or not self._should_retry(response):
            return response

        retry_system_prompt = (
            f"{system_prompt}\n"
            "Do not output <thought> tags, hidden reasoning, or any internal analysis.\n"
            "Write only the final five tags and their contents."
        )
        return self._generate_once(
            question=question,
            system_prompt=retry_system_prompt,
            assistant_prefix=self.generation_config.assistant_prefix,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
