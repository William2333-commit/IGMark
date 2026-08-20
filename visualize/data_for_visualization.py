# Code for anonymous submission
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================
# data_for_visualization.py
# Description: Data class for token-level watermark visualization.
# ==============================================


class DataForVisualization:
    """
    Container for token-level watermark visualization data.

    Attributes:
        decoded_tokens (list[str]): List of decoded token strings.
        highlight_values (list[int|float]): Per-token highlight values
            (e.g., green-flag: 1 = green, 0 = red, -1 = prefix).
        weight_values (list[float]): Per-token weight values
            (e.g., information gain score, -1 = prefix).
    """

    def __init__(
        self,
        decoded_tokens: list[str],
        highlight_values: list,
        weight_values: list,
    ) -> None:
        self.decoded_tokens = decoded_tokens
        self.highlight_values = highlight_values
        self.weight_values = weight_values

    def __len__(self) -> int:
        return len(self.decoded_tokens)

    def __repr__(self) -> str:
        return (
            f"DataForVisualization(tokens={len(self.decoded_tokens)}, "
            f"highlights={len(self.highlight_values)}, weights={len(self.weight_values)})"
        )
