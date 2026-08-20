"""HumanEvalPack dataset adapter for C++ and Java."""
from datasets import load_dataset
from evaluation.dataset import BaseDataset


class HEPDataset(BaseDataset):
    """
    HumanEvalPack dataset for C++/Java.

    Each sample:
      prompt: code prompt (ends with function/class signature)
      references: {'task': prompt, 'test': test_code, 'entry_point': function_name}
    """

    def __init__(self, language: str, max_samples: int = 200):
        super().__init__(max_samples=max_samples)
        self.language = language
        self.data = load_dataset("bigcode/humanevalpack", language, split="test")
        self.load_data()

    def load_data(self):
        for idx, item in enumerate(self.data):
            if idx >= self.max_samples:
                break
            prompt = item["prompt"]
            self.prompts.append(prompt)
            self.references.append({
                "task": prompt,
                "test": item["test"],
                "entry_point": item["entry_point"]
            })
