"""AST scope-safe variable renaming attack (LibCST-based).

Scope-safe: each local symbol's definition and all its references get the same
new name. Protected: function names, parameters, imports, builtins, attribute
names, dunders, global/nonlocal, underscore-prefixed (private).

Deterministic names var_001, var_002, ... (no WordNet dependency).
depth = number of renameable variables to rename in one pass.
"""
import random
import string
import keyword
import libcst as cst
import libcst.metadata as meta

# Builtins to protect
import builtins as _builtins
_BUILTINS = set(dir(_builtins)) | {"self", "cls"}
_KW = set(keyword.kwlist)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_builtin(name: str) -> bool:
    return name in _BUILTINS or name in _KW


def _random_identifier(rng: random.Random, used: set, min_len: int = 4, max_len: int = 8) -> str:
    """生成随机标识符(小写字母),避免关键字/builtin/已用名。"""
    while True:
        n = rng.randint(min_len, max_len)
        name = ''.join(rng.choice(string.ascii_lowercase) for _ in range(n))
        if name in used or _is_builtin(name) or _is_dunder(name):
            continue
        return name


class _Collector(cst.CSTVisitor):
    """Collect renameable local variables and protected names."""
    METADATA_DEPENDENCIES = (meta.ScopeProvider,)

    def __init__(self):
        self.param_names = set()
        self.func_names = set()
        self.imported_names = set()
        self.global_nonlocal_names = set()
        self.assign_keys = []  # list of (scope_id, name), dedup later
        self._seen = set()

    def _renameable(self, name: str) -> bool:
        if _is_dunder(name) or _is_builtin(name):
            return False
        if name in self.param_names or name in self.func_names:
            return False
        if name in self.imported_names or name in self.global_nonlocal_names:
            return False
        if name.startswith("_"):
            return False
        return True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.func_names.add(node.name.value)
        for p in node.params.params:
            self.param_names.add(p.name.value)
        for p in node.params.kwonly_params:
            self.param_names.add(p.name.value)
        if node.params.posonly_params:
            for p in node.params.posonly_params:
                self.param_names.add(p.name.value)
        if node.params.star_kwarg:
            self.param_names.add(node.params.star_kwarg.name.value)
        if node.params.star_arg and isinstance(node.params.star_arg, cst.Param):
            self.param_names.add(node.params.star_arg.name.value)

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            n = alias.asname.name.value if alias.asname else alias.name.name.value if hasattr(alias.name, 'name') else str(alias.name)
            self.imported_names.add(n)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.names, cst.ImportStar):
            return  # `from x import *`，无法提取具名导入
        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                n = alias.asname.name.value if alias.asname else (alias.name.name.value if hasattr(alias.name, 'name') else None)
                if n: self.imported_names.add(n)

    def visit_Global(self, node: cst.Global) -> None:
        for n in node.names:
            if hasattr(n, "value"):
                self.global_nonlocal_names.add(n.value)

    def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
        for n in node.names:
            if hasattr(n, "value"):
                self.global_nonlocal_names.add(n.value)

    def _extract_names(self, node):
        """Recursively extract Name nodes from a target (handles tuple unpacking)."""
        names = []
        if isinstance(node, cst.Name):
            names.append(node)
        elif isinstance(node, (cst.Tuple, cst.List)):
            for el in node.elements:
                child = el.value if hasattr(el, "value") else el
                names.extend(self._extract_names(child))
        return names

    def _record_target(self, target_node):
        for name_node in self._extract_names(target_node):
            name = name_node.value
            if not self._renameable(name):
                continue
            try:
                scope = self.get_metadata(meta.ScopeProvider, name_node, None)
            except KeyError:
                continue
            if scope is None:
                continue
            key = (id(scope), name)
            if key not in self._seen:
                self._seen.add(key)
                self.assign_keys.append(key)

    def visit_Assign(self, node: cst.Assign) -> None:
        for t in node.targets:
            self._record_target(t.target)

    def visit_For(self, node: cst.For) -> None:
        self._record_target(node.target)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if node.target:
            self._record_target(node.target)

    def visit_WithItem(self, node: cst.WithItem) -> None:
        if node.asname:
            self._record_target(node.asname.name if hasattr(node.asname, 'name') else node.asname)


class _Renamer(cst.CSTTransformer):
    """Replace Name nodes whose (scope, name) is in rename_map."""
    METADATA_DEPENDENCIES = (meta.ScopeProvider,)

    def __init__(self, rename_map):
        self.rename_map = rename_map  # {(scope_id, name): new_name}

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name):
        try:
            scope = self.get_metadata(meta.ScopeProvider, original_node, None)
        except KeyError:
            return updated_node
        if scope is None:
            return updated_node
        key = (id(scope), updated_node.value)
        if key in self.rename_map:
            return updated_node.with_changes(value=self.rename_map[key])
        return updated_node


class VariableRenameEditor:
    """Scope-safe variable renaming attack.

    Args:
        depth: number of renameable variables to rename (0 = no attack).
        seed: random seed for variable selection order.
    """

    def __init__(self, depth: int = 5, seed: int = 0):
        self.depth = depth
        self.seed = seed

    def edit(self, text: str, reference=None) -> str:
        if not text or not text.strip():
            return text
        try:
            module = cst.parse_module(text)
        except Exception:
            return text
        wrapper = cst.MetadataWrapper(module)
        collector = _Collector()
        wrapper.visit(collector)

        keys = collector.assign_keys
        if not keys or self.depth <= 0:
            return text

        rng = random.Random(self.seed)
        order = list(keys)
        rng.shuffle(order)
        selected = order[: self.depth]

        # 用随机词替换(复现原论文 Rename: random word),避免确定性 var_001 巧合绿名单
        rename_map = {}
        used = set()
        for k in selected:
            new_name = _random_identifier(rng, used)
            rename_map[k] = new_name
            used.add(new_name)
        renamer = _Renamer(rename_map)
        new_module = wrapper.visit(renamer)
        try:
            return new_module.code
        except Exception:
            return text


# Self-test
if __name__ == "__main__":
    code = '''from typing import List

def has_close_elements(numbers, threshold):
    result = False
    for i, first_number in enumerate(numbers):
        for j, second_number in enumerate(numbers):
            if i != j and abs(first_number - second_number) < threshold:
                result = True
                break
    return result
'''
    print("=== Original ===")
    print(code)
    for s in range(3):
        out = VariableRenameEditor(depth=5, seed=s).edit(code)
        print(f"=== depth=5 seed={s} ===")
        print(out)
