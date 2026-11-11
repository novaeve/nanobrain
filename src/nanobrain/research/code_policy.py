"""A restricted tensor-code contract, complemented by a separate OS sandbox.

This static gate is not advertised as a proof of Python sandbox security.
"""
from __future__ import annotations

import ast
import builtins
import types

IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "torch.optim"}
BUILTINS = {k: getattr(builtins, k) for k in ["range", "len", "float", "int", "bool", "min", "max", "sum", "abs", "enumerate", "zip", "super", "dict", "list", "tuple", "isinstance", "ValueError"]}
NN = {"Module", "Linear", "LayerNorm", "RMSNorm", "Dropout", "GELU", "SiLU", "ReLU", "Tanh", "Sigmoid", "Softmax", "Sequential", "ModuleList", "Parameter", "Identity", "Embedding", "Conv1d", "MultiheadAttention", "TransformerEncoderLayer"}
TORCH = {"Tensor", "tensor", "zeros", "ones", "empty", "rand", "randn", "zeros_like", "ones_like", "arange", "cat", "stack", "mean", "sum", "log", "exp", "sqrt", "abs", "clamp", "softmax", "sigmoid", "tanh", "matmul", "bmm", "where", "isfinite", "no_grad", "float32", "bfloat16", "long", "bool"}
FUNCTIONAL = {"gelu", "silu", "relu", "normalize", "linear", "layer_norm", "dropout", "softmax", "log_softmax", "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "smooth_l1_loss", "kl_div", "binary_cross_entropy_with_logits", "softplus"}
FORBIDDEN_ATTRS = {"save", "load", "storage", "untyped_storage", "data_ptr", "register_hook", "register_forward_hook", "register_backward_hook", "from_file", "set_", "resize_", "numpy", "tofile", "dump", "read", "write", "open", "read_text", "write_text", "format", "format_map"}


def dotted(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return dotted(node.value) + "." + node.attr
    return ""


def inspect_source(source, entrypoint):
    tree = ast.parse(source)
    if len(list(ast.walk(tree))) > 5000: raise ValueError("source AST exceeds 5000 nodes")
    functions = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if entrypoint not in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}:
        raise ValueError("plugin entrypoint must be a top-level function")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.Delete)):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_") and node.id != "__init__":
            raise ValueError("private/introspection names are forbidden")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and node.attr != "__init__": raise ValueError("private/introspection attributes are forbidden")
            if node.attr in FORBIDDEN_ATTRS: raise ValueError(f"forbidden attribute {node.attr}")
            name = dotted(node)
            if name.startswith("torch.") and name.split('.')[1] not in TORCH | {"nn", "optim"}:
                raise ValueError(f"torch API not allowed: {name}")
            if name.startswith("torch.nn."):
                suffix = name[len("torch.nn."):]
                if suffix not in NN | {"functional"} and not (suffix.startswith("functional.") and suffix.split('.')[1] in FUNCTIONAL and len(suffix.split('.')) == 2):
                    raise ValueError("unapproved qualified nn API")
            if name.startswith("torch.optim.") and name not in {"torch.optim.Optimizer", "torch.optim.SGD", "torch.optim.Adam", "torch.optim.AdamW", "torch.optim.RMSprop", "torch.optim.Adagrad"}:
                raise ValueError("unapproved optimizer API")
            if name.startswith("nn.") and name.split('.')[1] not in NN: raise ValueError(f"nn API not allowed: {name}")
            if name.startswith("F.") and name.split('.')[1] not in FUNCTIONAL: raise ValueError(f"functional API not allowed: {name}")
        if isinstance(node, ast.Import):
            if any(a.name not in IMPORTS or a.asname not in {None, "torch", "nn", "F"} for a in node.names): raise ValueError("only approved torch imports are allowed")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in IMPORTS: raise ValueError("only approved torch imports are allowed")
            allowed = NN if node.module == "torch.nn" else FUNCTIONAL if node.module == "torch.nn.functional" else {"Optimizer"} if node.module == "torch.optim" else TORCH | {"nn"}
            if any(a.name not in allowed for a in node.names): raise ValueError("unapproved imported symbol")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in BUILTINS and node.func.id not in functions and node.func.id not in NN | TORCH | FUNCTIONAL | {"Optimizer"}:
                raise ValueError(f"unapproved callable: {node.func.id}")
        if isinstance(node, ast.ClassDef):
            if len(node.bases) != 1 or dotted(node.bases[0]) not in {"nn.Module", "Optimizer", "torch.optim.Optimizer"}:
                raise ValueError("custom classes must derive directly from nn.Module or Optimizer")
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_") and node.name != "__init__": raise ValueError("private/special methods are forbidden")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if dotted(value) in {"torch", "nn", "F", "torch.nn", "torch.nn.functional", "torch.optim"}:
                raise ValueError("module aliases may only be introduced by approved imports")
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(dotted(t).startswith(("torch.", "nn.", "F.")) for t in targets):
                raise ValueError("mutating framework modules is forbidden")
    # Import-time work is declarations and approved imports only; no training or I/O.
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise ValueError("top-level execution is forbidden")
    return tree


def load_tensor_code(source, entrypoint, namespace):
    tree = inspect_source(source, entrypoint)
    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in IMPORTS: raise ImportError("import outside tensor-code API")
        return builtins.__import__(name, globals, locals, fromlist, level)
    module = types.ModuleType(namespace)
    module.__dict__["__builtins__"] = {**BUILTINS, "__build_class__": builtins.__build_class__, "__import__": importer}
    exec(compile(tree, f"<{namespace}>", "exec"), module.__dict__)
    return module.__dict__[entrypoint]
