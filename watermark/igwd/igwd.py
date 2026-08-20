# Code for anonymous submission
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================
# igwd.py
# Description: IGWD — Information Gain Weighted Detection
#              Generation: all-token watermarking (KGW-style)
#              Detection:  IG-weighted generalized z-score
# ==============================================

import math
import torch
from functools import partial

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class IGWDConfig(BaseConfig):
    """Config class for IGWD algorithm."""

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict["gamma"]
        self.delta = self.config_dict["delta"]
        self.hash_key = self.config_dict["hash_key"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]
        self.eps = self.config_dict.get("eps", 1e-12)

    @property
    def algorithm_name(self) -> str:
        return "IGWD"


class IGWDUtils:
    """Utility class for IGWD algorithm."""

    def __init__(self, config: IGWDConfig, *args, **kwargs):
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

    def calculate_green_mass(
        self,
        probs_t: torch.Tensor,
        greenlist_ids: torch.LongTensor
    ) -> float:
        return probs_t[greenlist_ids].sum().item()

    def calculate_ig_from_green_mass(self, g_t: float) -> float:
        """
        IG_t = D_KL(q_t || p_t)
             = q_G,t * delta - log(1 + (exp(delta)-1) * g_t)
        """
        alpha = math.exp(self.config.delta)
        z_t = 1.0 + (alpha - 1.0) * g_t
        q_g_t = (alpha * g_t) / max(z_t, self.config.eps)
        ig_t = q_g_t * self.config.delta - math.log(max(z_t, self.config.eps))
        return ig_t

    def calculate_probabilities(self, model, tokenized_text: torch.Tensor) -> list[torch.Tensor | None]:
        """
        Calculate p_t for each token position via one forward pass.
        Returns list aligned with tokenized_text: probs_list[idx] = p_t predicting token at idx.
        """
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            logits = output.logits[0]

        probs_list = [None]
        for idx in range(1, logits.shape[0]):
            probs_t = torch.softmax(logits[idx - 1], dim=-1)
            probs_t = torch.clamp(probs_t, min=self.config.eps)
            probs_list.append(probs_t)
        return probs_list

    def score_sequence(
        self,
        input_ids: torch.Tensor,
        probs_list: list[torch.Tensor | None],
    ) -> tuple[float, list[int], list[float]]:
        """
        Score sequence with IG-weighted generalized z-score.

        w_t = max(0, IG_t - min(IG))
        z = Σ w_t · (1_green - g_t) / √(Σ w_t² · g_t · (1-g_t))

        Returns:
            z_score, green_token_flags, ig_values (used as weights for visualization)
        """
        if len(input_ids) <= self.config.prefix_length:
            raise ValueError("Sequence too short to score after prefix_length.")

        n = len(input_ids)
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        g_values = [0.0 for _ in range(self.config.prefix_length)]
        ig_values = [-1.0 for _ in range(self.config.prefix_length)]

        # Single pass: collect green flag, g_t, IG_t per position
        for idx in range(self.config.prefix_length, n):
            probs_t = probs_list[idx]
            if probs_t is None:
                green_token_flags.append(0)
                g_values.append(0.0)
                ig_values.append(0.0)
                continue

            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            curr_token = input_ids[idx]
            is_green = 1 if (curr_token in greenlist_ids) else 0
            green_token_flags.append(is_green)

            g_t = self.calculate_green_mass(probs_t, greenlist_ids)
            g_values.append(g_t)
            ig_t = self.calculate_ig_from_green_mass(g_t)
            ig_values.append(ig_t)

        # Linear weights: w_t = max(0, IG_t - min(IG))
        valid_ig = [v for v in ig_values[self.config.prefix_length:] if v >= 0]
        if len(valid_ig) == 0:
            raise ValueError("Must have at least 1 valid position to score.")

        min_ig = min(valid_ig)
        weights = []
        for v in ig_values:
            if v < 0:
                weights.append(-1.0)
            else:
                weights.append(max(0.0, v - min_ig))

        # Fallback: if all weights zero, use uniform
        w_sq = sum(w * w for w in weights[self.config.prefix_length:])
        if w_sq <= self.config.eps:
            for i in range(self.config.prefix_length, n):
                if ig_values[i] >= 0:
                    weights[i] = 1.0

        # Weighted generalized z-score
        numer_sum = 0.0
        var_sum = 0.0
        for idx in range(self.config.prefix_length, n):
            w_t = weights[idx]
            if w_t <= 0 or probs_list[idx] is None:
                continue
            is_green = green_token_flags[idx]
            g_t = g_values[idx]
            numer_sum += w_t * (is_green - g_t)
            var_sum += w_t * w_t * g_t * (1.0 - g_t)

        if var_sum <= 0:
            raise ValueError("Variance sum is zero, cannot compute z-score.")

        z_score = numer_sum / math.sqrt(var_sum)
        return z_score, green_token_flags, ig_values


class IGWDLogitsProcessor(LogitsProcessor):
    """Logits processor: watermark ALL tokens (KGW-style, no selectivity)."""

    def __init__(self, config: IGWDConfig, utils: IGWDUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(
        self,
        scores: torch.FloatTensor,
        greenlist_token_ids: list[torch.LongTensor]
    ) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        return green_tokens_mask.bool()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = []
        for b_idx in range(input_ids.shape[0]):
            batched_greenlist_ids.append(self.utils.get_greenlist_ids(input_ids[b_idx]))

        green_tokens_mask = self._calc_greenlist_mask(scores, batched_greenlist_ids)
        scores[green_tokens_mask] = scores[green_tokens_mask] + self.config.delta
        return scores


class IGWD(BaseWatermark):
    """IGWD: Information Gain Weighted Detection.

    Generation: KGW-style all-token watermarking.
    Detection: IG-weighted generalized z-score.
    """

    def __init__(
        self,
        algorithm_config: str | IGWDConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = IGWDConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, IGWDConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or an IGWDConfig instance")

        self.utils = IGWDUtils(self.config)
        self.logits_processor = IGWDLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text (all-token watermarking, KGW-style)."""
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        encoded_prompt = self.config.generation_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.config.device)

        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)

        watermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text,
            skip_special_tokens=True
        )[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark using IG-weighted generalized z-score."""
        encoded_text = self.config.generation_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        probs_list = self.utils.calculate_probabilities(
            self.config.generation_model,
            encoded_text
        )

        z_score, _, _ = self.utils.score_sequence(encoded_text, probs_list)
        is_watermarked = z_score > self.config.z_threshold

        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""
        encoded_text = self.config.generation_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(self.config.generation_model.device)

        probs_list = self.utils.calculate_probabilities(
            self.config.generation_model,
            encoded_text
        )

        _, highlight_values, weights = self.utils.score_sequence(
            encoded_text,
            probs_list
        )

        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        return DataForVisualization(decoded_tokens, highlight_values, weights)
