# Code for anonymous submission
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

class AlgorithmNameMismatchError(ValueError):
    """Exception raised when the algorithm name does not match the expected name."""
    def __init__(self, expected, actual):
        message = f"Expected algorithm name: {expected}, but got {actual}."
        super().__init__(message)


class LengthMismatchError(Exception):
    """Exception raised when the expected and actual lengths do not match."""
    def __init__(self, expected, actual):
        message = f"Expected length: {expected}, but got {actual}."
        super().__init__(message)
