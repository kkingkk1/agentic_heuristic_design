"""从已接入任务材料读取函数合同，供多任务控制器使用。"""

import ast
from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from scripts import p0_model_probe as probe


ROOT = Path(__file__).resolve().parents[1]
TASKS = frozenset({"tsp_aco", "op_aco", "cvrp_aco", "mkp_aco", "bpp_offline_aco", "tsp_gls"})


@dataclass(frozen=True)
class TaskContract:
    name: str
    black_box: bool
    train_size: int
    direction: str
    description: str
    function_description: str
    signature: str
    parameters: tuple[tuple[str, str], ...]

    def task_prompt(self):
        """只提供任务、待设计模块、接口和对应提示条件的必要上下文。"""
        return f"{self.description}\n{self.function_description}\nFunction interface: {self.signature}"

    def output_prompt(self):
        return ("Return <thought>algorithm description</thought> followed by the complete executable "
                "Python module, either directly or enclosed in <code> and </code>. "
                f"The required function interface is {self.signature}\n"
                "Include import numpy as np. Do not use tools or read files. No other text or Markdown fences.")

    def parse_proposal(self, text):
        try:
            thought, code = probe.tagged(text, ("thought", "code"))
        except probe.ProbeError as tagged_error:
            # 一对 thought 标签后若直接跟完整模块，提取原文，不修改候选代码。
            match = re.fullmatch(r"\s*<thought>\s*(.*?)\s*</thought>\s*(.*?)\s*", text, flags=re.DOTALL)
            if not match or not all(value.strip() for value in match.groups()):
                raise tagged_error
            thought, code = (value.strip() for value in match.groups())
        try:
            module = ast.parse(code)
        except SyntaxError as exc:
            raise probe.ProbeError(f"候选代码语法错误：{exc}") from exc
        functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == "heuristics_v2"]
        if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
            raise probe.ProbeError("候选须定义且仅定义一个同步 heuristics_v2 函数。")
        function = functions[0]
        args = function.args
        actual = tuple((argument.arg, ast.unparse(argument.annotation) if argument.annotation else None)
                       for argument in args.args)
        if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults
                or actual != self.parameters or function.returns is None
                or ast.unparse(function.returns) != "np.ndarray"):
            raise probe.ProbeError(f"候选函数签名不符合 {self.signature}。")
        if not any(isinstance(node, ast.Import)
                   and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
                   for node in module.body):
            raise probe.ProbeError("完整模块须包含 import numpy as np。")

        def returns_value(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return False
            if isinstance(node, ast.Return):
                return node.value is not None
            return any(returns_value(child) for child in ast.iter_child_nodes(node))

        if not any(returns_value(statement) for statement in function.body):
            raise probe.ProbeError("候选函数缺少返回值。")
        try:
            compile(module, "<model-proposal>", "exec")
        except SyntaxError as exc:
            raise probe.ProbeError(f"候选代码无法编译：{exc}") from exc
        return {"thought": thought, "code": code}


def load_task_contract(name, *, black_box=False):
    if name not in TASKS or type(black_box) is not bool or (name == "tsp_gls" and black_box):
        raise ValueError("任务名称或提示条件无效。")
    stem = name + ("_black_box" if black_box else "")
    config_path = ROOT / "cfg" / "problem" / f"{stem}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("problem_name") != name
            or config.get("problem_type") != ("black_box" if black_box else "gls" if name == "tsp_gls" else "aco")
            or config.get("obj_type") not in ("min", "max")
            or type(config.get("problem_size")) is not int or config["problem_size"] <= 0
            or not isinstance(config.get("description"), str) or not config["description"].strip()):
        raise ValueError(f"任务配置无效：{config_path}")
    prompt_dir = ROOT / "prompts" / stem
    description = (prompt_dir / "func_desc.txt").read_text(encoding="utf-8").strip()
    signature = (prompt_dir / "func_signature.txt").read_text(encoding="utf-8").strip().replace("{version}", "2")
    try:
        parsed = ast.parse(signature + "\n    pass\n")
    except SyntaxError as exc:
        raise ValueError(f"任务签名无效：{signature}") from exc
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        raise ValueError(f"任务签名无效：{signature}")
    function = parsed.body[0]
    args = function.args
    if (not description or function.name != "heuristics_v2" or args.posonlyargs or args.vararg
            or args.kwonlyargs or args.kwarg or args.defaults or function.returns is None
            or ast.unparse(function.returns) != "np.ndarray" or not args.args
            or any(argument.annotation is None or ast.unparse(argument.annotation) not in ("np.ndarray", "int", "float")
                   for argument in args.args)):
        raise ValueError(f"任务签名或描述无效：{signature}")
    parameters = tuple((argument.arg, ast.unparse(argument.annotation)) for argument in args.args)
    return TaskContract(name, black_box, config["problem_size"], config["obj_type"],
                        config["description"].strip(), description, signature, parameters)
