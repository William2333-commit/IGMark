# Code for anonymous submission
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ===============================================================
# base.py
# Description: Base classes for watermark algorithms.
# ===============================================================

from typing import Union
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization


class BaseConfig:
    """Base configuration class for watermark algorithms."""

    def __init__(self, algorithm_config_path: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
        Initialize the base configuration.

        Parameters:
            algorithm_config_path (str): Path to the algorithm configuration file.
            transformers_config (TransformersConfig): Configuration for the transformers model.
            **kwargs: Additional parameters to override config values.
        """
        # Load config file
        self.config_dict = load_config_file(algorithm_config_path)

        # Update config with kwargs
        if kwargs:
            self.config_dict.update(kwargs)

        # Load model-related configurations
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.transformers_config = transformers_config

        # Initialize algorithm-specific parameters
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters. Should be overridden by subclasses."""
        raise NotImplementedError

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name. Should be overridden by subclasses."""
        raise NotImplementedError


class BaseWatermark:
    """Base class for all watermark algorithms."""

    def __init__(self, algorithm_config: str | BaseConfig, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        pass

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text from a prompt."""
        raise NotImplementedError

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked (baseline) text from a prompt."""
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        encoded_unwatermarked_text = self.config.generation_model.generate(
            **encoded_prompt, **self.config.gen_kwargs
        )
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_unwatermarked_text, skip_special_tokens=True
        )[0]
        return unwatermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[tuple, dict]:
        """Detect watermark in a given text."""
        raise NotImplementedError

    def get_data_for_visualize(self, text, *args, **kwargs) -> DataForVisualization:
        """Get data for token-level visualization."""
        raise NotImplementedError
