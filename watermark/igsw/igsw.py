# Code for anonymous submission
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================
# igsw.py
# Description: IGSW — Information Gain Selective Watermarking
#              Generation: dynamic δ via sigmoid(IG), all tokens watermarked
#              Detection:  generalized z-score on all tokens
# ==============================================

import math
import torch
from functools import partial

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class IGSWConfig(BaseConfig):
    """Config class for IGSW algorithm."""

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict["gamma"]
        self.delta_ref = self.config_dict["delta_ref"]
        self.delta_min = self.config_dict["delta_min"]
        self.delta_max = self.config_dict["delta_max"]
        self.k = self.config_dict["k"]
        self.c0 = self.config_dict["c0"]
        self.hash_key = self.config_dict["hash_key"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]

        self.function = self.config_dict.get("function", "sigmoid")
        self.eps = self.config_dict.get("eps", 1e-12)
        self.visualize_mode = self.config_dict.get("visualize_mode", "raw_ig")

    @property
    def algorithm_name(self) -> str:
        return "IGSW"


class IGSWUtils:
    """Utility class for IGSW algorithm."""

    def __init__(self, config: IGSWConfig, *args, **kwargs):
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

    def calculate_probabilities(self, model, tokenized_text: torch.Tensor) -> list[torch.Tensor | None]:
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            logits = output.logits[0]

        probs_list = [None]
        for idx in range(1, logits.shape[0]):
            probs_t = torch.softmax(logits[idx - 1], dim=-1)
            probs_t = torch.clamp(probs_t, min=self.config.eps)
            probs_list.append(probs_t)
        return probs_list

    def calculate_green_mass(
        self,
        probs_t: torch.Tensor,
        greenlist_ids: torch.LongTensor
    ) -> float:
        return probs_t[greenlist_ids].sum().item()

    def calculate_qg_from_green_mass(self, g_t: float) -> float:
        alpha = math.exp(self.config.delta_ref)
        z_t = 1.0 + (alpha - 1.0) * g_t
        return (alpha * g_t) / max(z_t, self.config.eps)

    def calculate_ig_from_green_mass(self, g_t: float) -> float:
        """
        IG_t = D_KL(q_t || p_t) at reference strength delta_ref.
        """
        alpha = math.exp(self.config.delta_ref)
        z_t = 1.0 + (alpha - 1.0) * g_t
        q_g_t = (alpha * g_t) / max(z_t, self.config.eps)
        ig_t = q_g_t * self.config.delta_ref - math.log(max(z_t, self.config.eps))
        return ig_t

    def dynamic_delta(self, ig_t: float) -> float:
        """
        Watermark strength function: IG -> [delta_min, delta_max].

        Supports seven functions:
          sigmoid:     d = d_min + (d_max-d_min) / (1 + exp(-k*(IG-c0)))
          linear:      d = d_min + (d_max-d_min) * clip((IG-c0+k/2)/k, 0, 1)
          step:        d = d_max if IG >= c0 else d_min
          tanh:        d = d_min + (d_max-d_min) * (tanh(k*(IG-c0))+1)/2
          exponential: d = d_min + (d_max-d_min) * (1 - exp(-max(0, k*(IG-c0))))
          logarithmic: d = d_min + (d_max-d_min) * clip(k*ln(1+max(0, IG-c0)), 0, 1)
          piecewise:   d = linear transition from d_min to d_max within width k around c0
        """
        fn = self.config.function
        lo, hi = self.config.delta_min, self.config.delta_max
        k, c0 = self.config.k, self.config.c0

        if fn == "step":
            return float(hi if ig_t >= c0 else lo)

        if fn == "linear":
            r = max(abs(k), 1e-6)
            t = (ig_t - c0 + r / 2.0) / r
            t = max(0.0, min(1.0, t))
            return float(lo + (hi - lo) * t)

        if fn == "tanh":
            t = (math.tanh(k * (ig_t - c0)) + 1.0) / 2.0
            return float(lo + (hi - lo) * t)

        if fn == "exponential":
            x = max(0.0, k * (ig_t - c0))
            t = 1.0 - math.exp(-x)
            return float(lo + (hi - lo) * t)

        if fn == "logarithmic":
            x = max(0.0, ig_t - c0)
            t = k * math.log(1.0 + x)
            t = max(0.0, min(1.0, t))
            return float(lo + (hi - lo) * t)

        if fn == "piecewise":
            w = max(abs(k), 1e-6)  # k used as transition band width
            lo_ig = c0 - w / 2.0
            hi_ig = c0 + w / 2.0
            if ig_t <= lo_ig:
                return float(lo)
            if ig_t >= hi_ig:
                return float(hi)
            t = (ig_t - lo_ig) / w
            return float(lo + (hi - lo) * t)

        # default: sigmoid
        s = 1.0 / (1.0 + math.exp(-k * (ig_t - c0)))
        return float(lo + (hi - lo) * s)

    def score_sequence(
        self,
        input_ids: torch.Tensor,
        probs_list: list[torch.Tensor | None],
    ) -> tuple[float, list[int], list[float]]:
        """
        Score all tokens with generalized z-score:

            z = Σ (1_green − g_t) / √(Σ g_t·(1−g_t))

        All positions after prefix_length contribute.
        """
        if len(input_ids) <= self.config.prefix_length:
            raise ValueError("Sequence too short to score after prefix_length.")

        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        ig_values = [-1.0 for _ in range(self.config.prefix_length)]

        numer_sum = 0.0
        var_sum = 0.0

        for idx in range(self.config.prefix_length, len(input_ids)):
            probs_t = probs_list[idx]

            if probs_t is None:
                green_token_flags.append(0)
                ig_values.append(0.0)
                continue

            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            curr_token = input_ids[idx]
            is_green = 1 if (curr_token in greenlist_ids) else 0
            green_token_flags.append(is_green)

            g_t = self.calculate_green_mass(probs_t, greenlist_ids)
            ig_t = self.calculate_ig_from_green_mass(g_t)

            if self.config.visualize_mode == "raw_ig":
                ig_values.append(float(ig_t))
            else:
                ig_values.append(float(ig_t))

            numer_sum += (is_green - g_t)
            var_sum += g_t * (1.0 - g_t)

        if var_sum <= 0:
            raise ValueError("Variance sum is zero, cannot compute z-score.")

        z_score = numer_sum / math.sqrt(var_sum)
        return z_score, green_token_flags, ig_values

    def calculate_position_stats(self, model, tokenized_text: torch.Tensor):
        probs_list = self.calculate_probabilities(model, tokenized_text)
        position_stats = []

        for idx in range(self.config.prefix_length, len(tokenized_text)):
            probs_t = probs_list[idx]
            if probs_t is None:
                continue

            prefix_ids = tokenized_text[:idx]
            greenlist_ids = self.get_greenlist_ids(prefix_ids)

            curr_token = tokenized_text[idx]
            is_green = 1 if (curr_token in greenlist_ids) else 0

            g_t = self.calculate_green_mass(probs_t, greenlist_ids)
            q_g_t = self.calculate_qg_from_green_mass(g_t)
            ig_t = self.calculate_ig_from_green_mass(g_t)
            delta_t = self.dynamic_delta(ig_t)

            position_stats.append({
                "idx": int(idx),
                "is_green": int(is_green),
                "g_t": float(g_t),
                "q_g_t": float(q_g_t),
                "ig_t": float(ig_t),
                "delta_t": float(delta_t),
            })

        return position_stats


class IGSWLogitsProcessor(LogitsProcessor):
    """
    Logits processor: watermark ALL tokens with dynamic δ = sigmoid(IG).
    """

    def __init__(self, config: IGSWConfig, utils: IGSWUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        raw_probs = torch.softmax(scores, dim=-1)

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])

            g_t = raw_probs[b_idx][greenlist_ids].sum().item()
            ig_t = self.utils.calculate_ig_from_green_mass(g_t)
            delta_t = self.utils.dynamic_delta(ig_t)

            scores[b_idx][greenlist_ids] += delta_t

        return scores


class IGSW(BaseWatermark):
    """IGSW: Information Gain Selective Watermarking with dynamic δ."""

    def __init__(
        self,
        algorithm_config: str | IGSWConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = IGSWConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, IGSWConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or an IGSWConfig instance")

        self.utils = IGSWUtils(self.config)
        self.logits_processor = IGSWLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
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

    def get_position_stats(self, text: str):
        encoded_text = self.config.generation_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        return self.utils.calculate_position_stats(
            self.config.generation_model,
            encoded_text
        )

    def get_data_for_visualization(self, text: str, *args, **kwargs):
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
