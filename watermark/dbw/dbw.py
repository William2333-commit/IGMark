# Code for anonymous submission
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
import torch
from functools import partial

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class DBWConfig(BaseConfig):
    """Config class for DBW algorithm."""

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict["gamma"]
        self.delta_min = self.config_dict["delta_min"]
        self.delta_max = self.config_dict["delta_max"]
        self.k = self.config_dict["k"]
        self.c0 = self.config_dict["c0"]
        self.hash_key = self.config_dict["hash_key"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]
        self.tau = self.config_dict.get("tau", 1.0)
        self.eps = self.config_dict.get("eps", 1e-12)

    @property
    def algorithm_name(self) -> str:
        return "DBW"


class DBWUtils:
    """Utility class for DBW algorithm."""

    def __init__(self, config: DBWConfig, *args, **kwargs):
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(
            self.config.vocab_size,
            device=input_ids.device,
            generator=self.rng
        )
        return vocab_permutation[:greenlist_size]

    def calculate_spike_entropy(self, probs: torch.Tensor) -> float:
        """SE = Σ_v p_v / (1 + τ·p_v)  — DBW Eq."""
        return float(torch.sum(probs / (1.0 + self.config.tau * probs)).detach().cpu())

    def dynamic_delta(self, se: float) -> float:
        """δ(SE) = δ_min + (δ_max − δ_min) / (1 + e^(−k(SE − c0)))"""
        s = 1.0 / (1.0 + math.exp(-self.config.k * (se - self.config.c0)))
        return self.config.delta_min + (self.config.delta_max - self.config.delta_min) * s

    def calculate_entropy_list(self, model, tokenized_text: torch.Tensor) -> list[float]:
        """Compute spike entropy for each token position."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            logits = output.logits[0]

        entropy_list = [0.0]  # placeholder for first token (no prediction)
        for idx in range(1, logits.shape[0]):
            probs = torch.softmax(logits[idx - 1], dim=-1)
            se = self.calculate_spike_entropy(probs)
            entropy_list.append(se)
        return entropy_list

    def score_sequence(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[float, list[int]]:
        """
        KGW-style standard z-test:
          z = (|s|_G − γ·T) / √(γ(1−γ)·T)
        """
        n = len(input_ids)
        if n <= self.config.prefix_length:
            raise ValueError("Sequence too short to score.")

        green_flags = [-1 for _ in range(self.config.prefix_length)]
        green_count = 0
        for idx in range(self.config.prefix_length, n):
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            curr_token = input_ids[idx]
            is_green = 1 if curr_token in greenlist_ids else 0
            green_flags.append(is_green)
            green_count += is_green

        T = n - self.config.prefix_length
        gamma = self.config.gamma
        numer = green_count - gamma * T
        denom = math.sqrt(gamma * (1.0 - gamma) * T)
        if denom <= self.config.eps:
            raise ValueError("Denominator too small for z-score computation.")

        z_score = numer / denom
        return z_score, green_flags


class DBWLogitsProcessor(LogitsProcessor):
    """Logits processor: dynamic δ based on spike entropy."""

    def __init__(self, config: DBWConfig, utils: DBWUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        raw_probs = torch.softmax(scores, dim=-1)

        for b_idx in range(input_ids.shape[0]):
            prefix_ids = input_ids[b_idx]
            greenlist_ids = self.utils.get_greenlist_ids(prefix_ids)
            se = self.utils.calculate_spike_entropy(raw_probs[b_idx])
            delta_t = self.utils.dynamic_delta(se)
            scores[b_idx][greenlist_ids] += delta_t

        return scores


class DBW(BaseWatermark):
    """DBW: Dynamic Bias Watermarking (Xu et al., PR 2026)."""

    def __init__(
        self,
        algorithm_config: str | DBWConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = DBWConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, DBWConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a DBWConfig instance")

        self.utils = DBWUtils(self.config)
        self.logits_processor = DBWLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        return self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text, skip_special_tokens=True
        )[0]

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        z_score, _ = self.utils.score_sequence(encoded_text)
        is_watermarked = z_score > self.config.z_threshold

        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.generation_model.device)

        _, green_flags = self.utils.score_sequence(encoded_text)

        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        weights = [float(f) if f >= 0 else -1.0 for f in green_flags]
        return DataForVisualization(decoded_tokens, green_flags, weights)
