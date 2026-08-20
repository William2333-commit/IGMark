import ast
from datasets import load_dataset
from evaluation.dataset import BaseDataset


class MBPPDataset(BaseDataset):
    """
    MBPP dataset adapter.

    We convert each sample into:
      prompt: text prompt to generation model
      references: {
          'task': prompt,
          'test': joined test code,
          'entry_point': function name
      }
    """

    def __init__(self, split: str = "test", max_samples: int = 200, use_few_shot: bool = True):
        super().__init__(max_samples=max_samples)
        self.split = split
        self.use_few_shot = use_few_shot
        self.data = load_dataset("mbpp", split=split)
        self.load_data()

    def _extract_entry_point_from_code(self, code: str) -> str:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
        raise ValueError("Cannot find function name in MBPP reference code.")

    def _extract_signature(self, code: str) -> str:
        """Extract function signature line(s) from code, up to the closing colon."""
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Get source lines for the signature
                lines = code.split("\n")
                sig_end = node.body[0].lineno - 1 if node.body else node.end_lineno
                # join lines from function def up to the colon line
                sig_lines = []
                for i in range(node.lineno - 1, sig_end):
                    ln = lines[i]
                    sig_lines.append(ln)
                    if ":" in ln:
                        break
                return "\n".join(sig_lines)
        raise ValueError("Cannot extract function signature.")

    def _build_few_shot_prefix(self, examples):
        pieces = []
        for ex in examples:
            desc = ex["text"].strip()
            code = ex["code"].strip()
            pieces.append(
                f'You are an expert Python programmer.\n'
                f'Write a Python function for the following task.\n\n'
                f'Task:\n{desc}\n\n'
                f'Answer:\n{code}\n'
            )
        return "\n\n".join(pieces)

    def load_data(self):
        # MBPP official split usually contains prompt-like `text`, `code`, `test_list`
        # We'll optionally use the first 3 training examples as demonstrations.
        few_shot_prefix = ""
        if self.use_few_shot:
            train_data = load_dataset("mbpp", split="train")
            demos = [train_data[i] for i in range(3)]
            few_shot_prefix = self._build_few_shot_prefix(demos) + "\n\n"

        for idx, item in enumerate(self.data):
            if idx >= self.max_samples:
                break

            task_desc = item["text"].strip()
            canonical_code = item["code"].strip()
            test_list = item["test_list"]

            entry_point = self._extract_entry_point_from_code(canonical_code)
            signature = self._extract_signature(canonical_code)

            # Only give the function signature, NOT the body
            prompt = (
                few_shot_prefix
                + "You are an expert Python programmer.\n"
                + "Write a Python function for the following task.\n\n"
                + f"Task:\n{task_desc}\n\n"
                + f"Answer:\n{signature}\n"
            )

            # Wrap MBPP test assertions into HumanEval-compatible format:
            #   def check(candidate):
            #       assert candidate(args) == expected
            # The original assert statements call entry_point by name;
            # we replace that with 'candidate' so check(entry_point) works.
            wrapped_lines = []
            for tline in test_list:
                tline_stripped = tline.strip()
                if tline_stripped.startswith("assert "):
                    # Replace function name with 'candidate'
                    tline_stripped = tline_stripped.replace(entry_point + "(", "candidate(")
                wrapped_lines.append("    " + tline_stripped)
            test_code = "def check(candidate):\n" + "\n".join(wrapped_lines)

            # task provides the function signature so exec() sees a complete function.
            # After truncation the generated text is only the body (without "def ...").
            self.prompts.append(prompt)
            self.references.append({
                "task": f"{signature}\n",
                "test": test_code,
                "entry_point": entry_point
            })