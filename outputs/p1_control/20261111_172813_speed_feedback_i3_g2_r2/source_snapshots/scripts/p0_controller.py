"""P0 一代控制：角色输入、候选来源、失败回退和固定候选池。"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
import random
import re
import threading
import time
from typing import Any

from scripts import p0_model_probe as probe


TASK = probe.TASK
OUTPUT = ("Return exactly <thought>algorithm description</thought> followed by "
          "<code>the complete executable Python module, including import numpy as np "
          "and heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray</code>. "
          "Do not use tools or read files. No other text or Markdown fences.")


class StopRun(Exception):
    """无法可信继续本次运行。"""


@dataclass
class Candidate:
    candidate_id: str
    sequence: int
    stage: str
    parents: tuple[str, ...]
    thought: str = ""
    code: str = ""
    objective: float | None = None
    fitness: float | None = None
    generation_order: int | None = None
    completed_order: int | None = None
    evaluation_origin: str | None = None
    error: str | None = None
    raw_objective: float | None = None
    timing_seconds: float | None = None
    timing_scope: str | None = None

    @property
    def valid(self):
        return self.fitness is not None


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def check_modification_scope(code: str) -> None:
    """拒绝可静态识别的文件、进程和动态执行入口；这不是沙箱。"""
    tree = ast.parse(code)
    forbidden_modules = {"os", "sys", "subprocess", "pathlib", "shutil", "ctypes", "multiprocessing",
                         "socket", "importlib", "builtins", "pickle", "shelve"}
    forbidden_names = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                       "getattr", "setattr", "delattr", "globals", "locals", "vars", "__builtins__"}
    forbidden_attrs = {"save", "savez", "savez_compressed", "load", "fromfile", "tofile", "memmap",
                       "dump", "dumps", "loadtxt", "savetxt", "write_text", "write_bytes", "unlink",
                       "remove", "rename", "replace", "system", "popen", "spawn", "fork"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
            if any(name in forbidden_modules for name in modules):
                raise ValueError("候选含不允许的文件、进程或动态执行模块。")
        if isinstance(node, ast.ImportFrom):
            if ((node.module or "").split(".")[0] in forbidden_modules
                    or any(alias.name in forbidden_attrs or alias.name == "*" for alias in node.names)):
                raise ValueError("候选含不允许的文件、进程或动态执行导入。")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError("候选引用不允许的文件或动态执行入口。")
        if isinstance(node, ast.Attribute) and (node.attr in forbidden_attrs or node.attr.startswith("__")):
            raise ValueError("候选引用不允许的文件、进程或动态执行入口。")


DESIGN_ROLES = ("solution_quality_agent", "speed_agent")
INTEGRATOR_ROLE = "integrator"
ROLE_DETAILS = {
    "solution_quality_agent": {
        "label": "Solution Quality Agent",
        "focus": "improving final-solution quality according to the task objective",
        "identity": "Act as the Solution Quality Agent.\n",
        "opening": "Here is an algorithm proposed by previous agents:",
        "direction": (
            "focusing on {focus}. Propose a change to the editable heuristic that may improve the "
            "solver's final solutions in the stated task direction. The heuristic's returned values "
            "guide the solver; they are not the final objective. Preserve feasibility, the required "
            "interface, the task objective, solver, evaluator and stopping rules. Use the critic's "
            "feedback as hypotheses, not proof of causation. A combined score does not separately "
            "establish quality or runtime changes; distinguish expected effects from supplied observations."
        ),
    },
    "speed_agent": {
        "label": "Speed Agent",
        "focus": "improving final-solution quality benefit per complete solver runtime",
        "identity": "Act as the Speed Agent.\n",
        "opening": "Here are some algorithms proposed by previous agents:",
        "direction": (
            "to refine and improve it through {focus}. Change only the editable heuristic: an equivalent "
            "implementation or a different guidance rule is valid. Explain how it may improve final "
            "solution benefit per complete solve time, including possible quality and runtime effects. "
            "Shorter code or a faster heuristic alone does not "
            "prove a faster complete solver; model response time is a different cost. Preserve feasibility, "
            "the required interface, the task objective, solver, evaluator and stopping rules. Treat the "
            "critic's explanations as hypotheses. A combined score does not separately establish runtime "
            "or quality changes; do not invent measurements or claim quality preservation without evidence."
        ),
    },
}


def role_prompt(role: str, candidate: Candidate, critic: dict[str, str], *, task: str = TASK,
                output: str = OUTPUT, details: dict[str, dict[str, str]] = ROLE_DETAILS) -> str:
    if role not in details:
        raise ValueError("未知设计角色。")
    spec = details[role]
    direction = spec["direction"].format(focus=spec["focus"])
    return (f"{task}\n{spec['identity']}{spec['opening']}\nAlgorithm description: {candidate.thought}\n"
            f"Code: {candidate.code}\nObjective score: {candidate.objective}\n"
            f"Critic's feedback: {critic['advice']}\nReflection: {critic['reflection']}\n"
            "Based on the critic's feedback and the reflection, please propose a new algorithm "
            f"that improves the previous ones {direction}\n{output}")


def round_critic_prompt(first: Candidate, second: Candidate, *, task: str = TASK, direction: str = "min") -> str:
    better_direction = "lower" if direction == "min" else "higher"
    if second.fitness == first.fitness:
        comparison = "The objective scores are tied. Do not claim either score is better."
    else:
        better = "A" if first.fitness > second.fitness else "B"
        comparison = f"Candidate {better} has the better score. A {better_direction} objective score is better."
    # 论文未展开 invalid_ind_prompt；已确认合同只记录技术错误，不向角色追加错误报告。
    return (f"{task}\nAs an optimization expert in the domain of heuristic optimization, compare:\n"
            f"Candidate A algorithm description: {first.thought}\nCandidate A code: {first.code}\n"
            f"Candidate A objective score: {first.objective}\n"
            f"Candidate B algorithm description: {second.thought}\nCandidate B code: {second.code}\n"
            f"Candidate B objective score: {second.objective}\n\n{comparison}\n"
            "These candidates are from separate branches; do not assume either is the other's parent. "
            "Compare their designs, treat causal explanations as hypotheses, and give targeted advice.\n"
            "Format your response as:\n<ref>Your reflection on the candidate designs</ref> "
            "<ans>Specific critique and actionable suggestions (max 40 words)</ans>\n"
            "Let's think step by step and be constructive in your analysis.")


def parse_round_critic(message: str) -> dict[str, str]:
    reflection, advice = probe.tagged(message, ("ref", "ans"))
    if probe.english_words(advice) > 40:
        raise probe.ProbeError("逐轮 Critic 建议超过 40 词。")
    return {"reflection": reflection, "advice": advice}


SOLVER_TIME_SCOPE = "solve_wall_seconds_including_preparation_and_validation"


def read_solve_timing(start: dict[str, Any], instances: list[dict[str, Any]],
                      complete: dict[str, Any]) -> dict[str, Any]:
    """读取逐实例计时；兼容没有计时字段的历史记录，但不接受不完整的计时。"""
    scope = start.get("time_scope")
    if scope is None:
        if ("average_solve_wall_seconds" in complete
                or any("solve_wall_seconds" in instance for instance in instances)):
            raise ValueError("求解时间缺少计时范围。")
        return {}
    if scope != SOLVER_TIME_SCOPE or not instances:
        raise ValueError("求解计时范围或实例数量无效。")
    times = []
    for index, instance in enumerate(instances):
        value = instance.get("solve_wall_seconds")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"实例 {index} 的完整求解时间无效。")
        times.append(float(value))
    average = math.fsum(times) / len(times)
    reported = complete.get("average_solve_wall_seconds")
    if (isinstance(reported, bool) or not isinstance(reported, (int, float))
            or not math.isfinite(reported)
            or not math.isclose(reported, average, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("平均求解时间与逐实例结果不一致。")
    return {"solve_wall_seconds": times, "average_solve_wall_seconds": average}


def timing_from_result(result: dict[str, Any]) -> tuple[float | None, str | None]:
    """只接收评价器验证过的逐实例完整求解时间。"""
    if "average_solve_wall_seconds" in result:
        value = result["average_solve_wall_seconds"]
        start = result.get("start")
        if not isinstance(start, dict) or start.get("time_scope") != SOLVER_TIME_SCOPE:
            raise StopRun("平均求解耗时缺少已验证的计时范围。")
        scope = SOLVER_TIME_SCOPE
    else:
        return None, None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise StopRun("评价返回的耗时证据无效。")
    return float(value), scope


def timing_description(candidate: Candidate) -> str:
    """描述实测的平均完整求解时间及其范围。"""
    value = candidate.timing_seconds
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise StopRun(f"候选 {candidate.candidate_id} 缺少可信的耗时证据。")
    if candidate.timing_scope == SOLVER_TIME_SCOPE:
        return (f"mean complete solver wall time: {value} seconds per instance "
                "(includes preparation and validation; excludes process startup and model calls)")
    raise StopRun(f"候选 {candidate.candidate_id} 的耗时范围不可信。")


def speed_feedback_metrics(candidate: Candidate, raw_direction: str) -> tuple[float, float]:
    """仅为角色反馈计算最终质量收益及其与平均完整求解时间之比。"""
    timing_description(candidate)
    raw = candidate.raw_objective
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise StopRun(f"候选 {candidate.candidate_id} 缺少可信的原始目标值。")
    if raw_direction == "min":
        if raw <= 0:
            raise StopRun(f"候选 {candidate.candidate_id} 的最小化目标必须为正，才能取倒数。")
        benefit = 1.0 / raw
        product = raw * candidate.timing_seconds
        if not math.isfinite(product) or product <= 0:
            raise StopRun(f"候选 {candidate.candidate_id} 的目标与耗时乘积无效。")
        rate = 1.0 / product
    elif raw_direction == "max":
        if raw < 0:
            raise StopRun(f"候选 {candidate.candidate_id} 的最大化收益不能为负。")
        benefit = float(raw)
        rate = benefit / candidate.timing_seconds
    else:
        raise StopRun("速度反馈的原始目标方向无效。")
    if not math.isfinite(benefit) or not math.isfinite(rate):
        raise StopRun(f"候选 {candidate.candidate_id} 的质量收益或收益时间比不是有限数。")
    return benefit, rate


def feedback_stop_description(evaluation_mode: str) -> str:
    if evaluation_mode == "final_quality":
        return "The result is measured at the fixed iteration budget; this does not prove convergence."
    if evaluation_mode == "quality_per_time":
        return "The result is measured at stagnation or the iteration limit; neither proves optimality."
    raise StopRun("速度反馈的评价设置无效。")


def split_critic_prompt(first: Candidate, second: Candidate, *, task: str, direction: str,
                        raw_direction: str, initial: bool = False, evaluation_mode: str = "final_quality") -> str:
    """一次比较产生两条互相隔离的角色反馈，并比较质量收益与完整求解时间之比。"""
    if first.raw_objective is None or second.raw_objective is None:
        raise StopRun("Critic 缺少候选的原始目标值。")
    first_time, second_time = timing_description(first), timing_description(second)
    first_benefit, first_rate = speed_feedback_metrics(first, raw_direction)
    second_benefit, second_rate = speed_feedback_metrics(second, raw_direction)

    def compare(first_value: float, second_value: float, *, lower_is_better: bool) -> str:
        if first_value == second_value:
            return "tie"
        return "A" if (first_value < second_value) == lower_is_better else "B"

    fitness_winner = compare(first.fitness, second.fitness, lower_is_better=False)
    raw_winner = compare(first.raw_objective, second.raw_objective, lower_is_better=raw_direction == "min")
    time_winner = compare(first.timing_seconds, second.timing_seconds, lower_is_better=True)
    rate_winner = compare(first_rate, second_rate, lower_is_better=False)
    better_direction = "lower" if direction == "min" else "higher"
    if first.fitness == second.fitness:
        comparison = "The objective scores are tied. Do not claim either score is better."
    else:
        better = "A" if first.fitness > second.fitness else "B"
        comparison = (f"Candidate {better} has the better score. "
                      f"A {better_direction} objective score is better.")
    targets = ("Both branches start from Candidate B's current code, not Candidate A's code. "
               "Quality advice target: Candidate B. Speed advice target: Candidate B."
               if initial else
               "Candidate A is the Solution Quality Agent's current candidate; Candidate B is the Speed Agent's "
               "current candidate. Quality advice target: Candidate A. Speed advice target: Candidate B.")
    raw_better = "lower" if raw_direction == "min" else "higher"
    benefit_formula = "1 / mean raw objective" if raw_direction == "min" else "mean raw objective"
    return (f"{task}\nAs an optimization expert, compare two evaluated candidates.\n"
            f"Candidate A algorithm description: {first.thought}\nCandidate A code: {first.code}\n"
            f"Candidate A objective score: {first.objective}; raw task objective: {first.raw_objective}; "
            f"Fitness: {first.fitness}; measured timing: {first_time}; "
            f"final-solution benefit: {first_benefit}; benefit per solve second: {first_rate}.\n"
            f"Candidate B algorithm description: {second.thought}\nCandidate B code: {second.code}\n"
            f"Candidate B objective score: {second.objective}; raw task objective: {second.raw_objective}; "
            f"Fitness: {second.fitness}; measured timing: {second_time}; "
            f"final-solution benefit: {second_benefit}; benefit per solve second: {second_rate}.\n"
            f"{comparison} When comparing raw task objectives, a {raw_better} raw task objective is better.\n"
            "The following comparisons are computed from the supplied measurements; A and B name candidates, "
            "'tie' means equal. Do not contradict them or conflate the metrics:\n"
            f"Fitness comparison: {fitness_winner} (higher Fitness is better).\n"
            f"Raw objective comparison: {raw_winner} ({raw_better} raw objective is better).\n"
            f"Observed solve time comparison: {time_winner} (lower recorded mean is shorter).\n"
            f"Benefit per solve second comparison: {rate_winner} (higher is better for speed advice). "
            f"Benefit = {benefit_formula}; rate = benefit / mean complete solver wall time. "
            f"{feedback_stop_description(evaluation_mode)}\n"
            f"{targets}\n"
            "Keep each reflection and advice focused on its assigned role; each branch receives only its own tags. "
            "Each recipient must be able to follow its advice without access to the other candidate's code. "
            "You may transfer an idea from the other candidate, but express it as a self-contained, specific "
            "change to the recipient's target code; do not ask a recipient to preserve, copy or modify unseen code. "
            "Each advice must propose a concrete change the recipient can make now. Do not ask the agents to "
            "run tests, repeat evaluations, benchmark, profile, use tools or read files; the controller evaluates "
            "the proposals. State uncertainty in the reflection when evidence is insufficient. "
            "Base speed advice on the supplied benefit per solve second, considering both final quality "
            "and measured timing; shorter time alone is not a better speed result. Timing is from a single evaluation "
            "and may be affected by concurrent load; compare only matching scopes. Treat causal explanations and "
            "unverified improvements as hypotheses; do not invent measurements. Do not assume the two "
            "candidates have a parent-child relationship.\n"
            "Return exactly these four nonempty tags, in order:\n"
            "<quality_ref>Quality reflection</quality_ref><quality_ans>Quality advice, max 40 words</quality_ans>"
            "<speed_ref>Speed reflection</speed_ref><speed_ans>Speed advice, max 40 words</speed_ans>")


def parse_split_critic(message: str) -> dict[str, dict[str, str]]:
    names = ("quality_ref", "quality_ans", "speed_ref", "speed_ans")
    quality_ref, quality_ans, speed_ref, speed_ans = probe.tagged(message, names)
    if any(re.search(r"</?(?:quality_ref|quality_ans|speed_ref|speed_ans)>", value)
           for value in (quality_ref, quality_ans, speed_ref, speed_ans)):
        raise probe.ProbeError("Critic 的角色反馈标签不能嵌套。")
    if probe.english_words(quality_ans) > 40 or probe.english_words(speed_ans) > 40:
        raise probe.ProbeError("Critic 的单角色建议超过 40 词。")
    return {
        "solution_quality_agent": {"reflection": quality_ref, "advice": quality_ans},
        "speed_agent": {"reflection": speed_ref, "advice": speed_ans},
    }


class Controller:
    """backend 提供 model、evaluate、before_generation；log 提供 event。"""

    design_roles = DESIGN_ROLES
    role_details = ROLE_DETAILS
    integrator_role = INTEGRATOR_ROLE
    role_specific_critic = True
    fusion_wording = {
        "first_final_label": "Solution Quality Agent", "second_final_label": "Speed Agent",
        "round_priorities": "the solution quality contribution of the Solution Quality Agent, the computational efficiency contribution of the Speed Agent",
        "balance": "solution quality and computational efficiency",
        "first_final_priority": "solution quality improvement potential from the best Solution Quality Agent",
        "second_final_priority": "computational efficiency improvement capabilities from the best Speed Agent",
    }

    def __init__(self, backend, log, *, population_size=10, initial_count=30, rounds=3, seed=20261103,
                 task_contract=None, generations=1):
        self.backend, self.log = backend, log
        self.n, self.initial_count, self.rounds = population_size, initial_count, rounds
        if type(generations) is not int or generations < 1:
            raise ValueError("演化代数必须为正整数。")
        self.generations, self.generation = generations, 0
        self.task = TASK if task_contract is None else task_contract.task_prompt()
        self.output = OUTPUT if task_contract is None else task_contract.output_prompt()
        self.direction = "min" if task_contract is None else task_contract.direction
        self.raw_direction = self.direction
        config = getattr(backend, "cfg", {})
        self.evaluation_mode = config.get("task", {}).get("evaluation_mode", "final_quality")
        if self.evaluation_mode == "quality_per_time":
            self.direction = "max"
            self.task += ("\nEvaluation objective score: 1 / (mean final tour length * mean complete "
                          "solver wall time). Higher scores are better. All candidates use the same stopping rule.")
        elif self.evaluation_mode != "final_quality":
            raise ValueError("未知评价设置。")
        if self.direction not in ("min", "max"):
            raise ValueError("目标方向必须为 min 或 max。")
        self.parse_proposal = probe.parse_proposal if task_contract is None else task_contract.parse_proposal
        self.random = random.Random(seed)
        self.candidates: list[Candidate] = []
        self._lock = threading.Lock()
        self._generated = 0
        self.stop = threading.Event()
        self.history: dict[str, list[str]] = {role: [] for role in (*self.design_roles, self.integrator_role)}

    def event(self, kind: str, **fields: Any) -> None:
        self.log.event(kind, **fields)

    def request(self, stage: str, prompt: str, parser):
        if self.stop.is_set():
            raise StopRun("已有并发任务停止本次运行。")
        try:
            response = self.backend.model(stage, prompt)
            self.event("model_response", stage=stage, generation=self.generation, prompt=prompt, response=response)
            return parser(response)
        except probe.ProbeError as exc:
            self.event("invalid_role_response", stage=stage, generation=self.generation, error=str(exc))
            self.stop.set()
            raise StopRun(f"{stage} 的角色输出无效：{exc}") from exc
        except StopRun:
            self.stop.set()
            raise

    def register(self, stage: str, parents: tuple[str, ...]) -> Candidate:
        with self._lock:
            sequence = len(self.candidates) + 1
            candidate = Candidate(f"c{sequence:04d}", sequence, stage, parents)
            self.candidates.append(candidate)
        parts = stage.split("_", 2)
        if parts[0] == "init":
            role, operator, generation, round_number = "initializer", "initialization", 0, None
        elif parts[0] == "eoh":
            role, operator, generation, round_number = "EoH", parts[1], self.generation, None
        elif parts[0] == "start":
            role, operator, generation, round_number = stage.removeprefix("start_"), "startup", self.generation, 0
        elif parts[0] == "round":
            role, operator, generation, round_number = parts[2], "continuation", self.generation, int(parts[1])
        elif parts[0] == "fusion":
            role, operator, generation = self.integrator_role, "elite_fusion" if parts[1] == "final" else "round_fusion", self.generation
            round_number = None if parts[1] == "final" else int(parts[1])
        else:
            role, operator, generation, round_number = parts[2], "elite_mutation", self.generation, None
        self.event("candidate_registered", candidate_id=candidate.candidate_id, sequence=sequence,
                   stage=stage, source_role=role, operator=operator, generation=generation,
                   round=round_number, parents=list(parents))
        return candidate

    def propose(self, candidate: Candidate, prompt: str):
        if self.stop.is_set():
            self.event("candidate_skipped", candidate_id=candidate.candidate_id, reason="run_stopped")
            return candidate
        reserved = False
        try:
            self.backend.prepare_proposal(candidate)
            reserved = True
            response = self.backend.model(candidate.stage, prompt)
            with self._lock:
                self._generated += 1
                candidate.generation_order = self._generated
                self.event("proposal_response", candidate_id=candidate.candidate_id, stage=candidate.stage,
                           generation=self.generation,
                           generation_order=candidate.generation_order, prompt=prompt, response=response)
            try:
                parsed = self.parse_proposal(response)
                candidate.thought, candidate.code = parsed["thought"], parsed["code"]
                check_modification_scope(candidate.code)
            except (probe.ProbeError, ValueError, SyntaxError) as exc:
                candidate.error = f"invalid_proposal: {exc}"
                self.event("candidate_invalid", candidate_id=candidate.candidate_id, error=candidate.error)
                return candidate
            result = self.backend.evaluate(candidate)
            self.event("evaluation", candidate_id=candidate.candidate_id, result=result)
            if result.get("status") == "infrastructure_error":
                raise StopRun(f"评价基础设施不可信：{result.get('error')}")
            if result.get("status") != "passed":
                candidate.error = str(result.get("error") or result.get("status"))
                return candidate
            objective, fitness = result.get("average"), result.get("fitness")
            if (not isinstance(objective, (int, float)) or isinstance(objective, bool)
                    or not isinstance(fitness, (int, float)) or isinstance(fitness, bool)):
                raise StopRun("评价器返回了无效分数类型。")
            if self.evaluation_mode == "quality_per_time":
                elapsed = result.get("average_solve_wall_seconds")
                if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                        or not math.isfinite(elapsed) or elapsed <= 0 or objective <= 0):
                    raise StopRun("速度评价返回无效的平均求解时间或路线长度。")
                expected_fitness = 1.0 / (objective * elapsed)
            else:
                expected_fitness = -objective if self.direction == "min" else objective
            if not math.isfinite(objective) or not math.isfinite(fitness) or fitness != expected_fitness:
                raise StopRun("评价器返回的目标或 Fitness 不可信。")
            if self.role_specific_critic:
                candidate.raw_objective = float(objective)
                candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
            with self._lock:
                candidate.completed_order = result.get("completed_monotonic_ns", time.monotonic_ns())
            candidate.objective = float(fitness if self.evaluation_mode == "quality_per_time" else objective)
            candidate.fitness = float(fitness)
            candidate.evaluation_origin = candidate.candidate_id
            return candidate
        except StopRun:
            self.stop.set()
            cancelled = getattr(self.backend, "cancelled", None)
            if cancelled is not None:
                cancelled.set()
            raise
        except Exception as exc:
            self.stop.set()
            cancelled = getattr(self.backend, "cancelled", None)
            if cancelled is not None:
                cancelled.set()
            self.event("candidate_infrastructure_error", candidate_id=candidate.candidate_id,
                       error=f"{type(exc).__name__}: {exc}")
            raise StopRun(f"候选处理意外失败：{type(exc).__name__}: {exc}") from exc
        finally:
            if reserved:
                self.backend.release_proposal(candidate)

    def batch(self, specs):
        entries = [(self.register(stage, tuple(parent.candidate_id for parent in parents)), prompt)
                   for stage, parents, prompt in specs]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.propose, candidate, prompt) for candidate, prompt in entries]
            results, failures = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    self.stop.set()
                    failures.append(exc)
            if failures:
                raise StopRun(f"并发阶段停止：{failures[0]}") from failures[0]
            return results

    def select(self, candidates, needed):
        valid = [c for c in candidates if c is not None and c.valid]
        all_valid = [c for c in self.candidates if c.valid]
        if any(item.generation_order is None for item in all_valid):
            raise StopRun("有效候选缺少实际生成顺序。")
        earliest = {}
        for item in sorted(all_valid, key=lambda c: c.completed_order):
            earliest.setdefault(code_hash(item.code), item)
        unique = {}
        for item in sorted(valid, key=lambda c: c.generation_order):
            key = code_hash(item.code)
            origin = earliest[key]
            unique.setdefault(key, replace(item, objective=origin.objective, fitness=origin.fitness,
                                           raw_objective=origin.raw_objective,
                                           timing_seconds=origin.timing_seconds, timing_scope=origin.timing_scope,
                                           evaluation_origin=origin.candidate_id))
        selected = sorted(unique.values(), key=lambda c: (-c.fitness, c.generation_order))
        if len(selected) < needed:
            raise StopRun(f"有效不同候选仅 {len(selected)} 个，少于所需 {needed} 个。")
        return selected[:needed]

    def eoh_parents(self, population, count):
        pool = list(population)
        parents = []
        for _ in range(count):
            weights = [1 / (population.index(candidate) + 1 + len(population)) for candidate in pool]
            chosen = self.random.choices(pool, weights=weights, k=1)[0]
            pool.remove(chosen)
            parents.append(chosen)
        return parents

    def elite_pair(self, population):
        if len(population) < 2:
            raise StopRun("种群不足两个候选。")
        # 排除无合法后继的排名，等价于对原分布进行条件重采样。
        first = self.random.choices(range(len(population) - 1),
                                    weights=[(i + 1) ** -3 for i in range(len(population) - 1)], k=1)[0]
        second = self.random.choice([i for i in (first + 1, first + 2) if i < len(population)])
        return population[first], population[second]

    def role_focus(self, role):
        return self.role_details[role]["focus"]

    def branch_prompt(self, role, candidate, critic):
        if self.role_specific_critic and role == "speed_agent":
            benefit, rate = speed_feedback_metrics(candidate, self.raw_direction)
            benefit_formula = "1 / mean raw objective" if self.raw_direction == "min" else "mean raw objective"
            critic = {**critic, "advice": f"{critic['advice']}\n"
                      f"Your current candidate's mean raw objective: {candidate.raw_objective}; "
                      f"final-solution benefit ({benefit_formula}): {benefit}; "
                      f"{timing_description(candidate)}; benefit per solve second: {rate} (higher is better). "
                      f"{feedback_stop_description(self.evaluation_mode)} "
                      "A single measurement can be affected by concurrent load."}
        return role_prompt(role, candidate, critic, task=self.task, output=self.output, details=self.role_details)

    def branch_critic_prompt(self, first, second):
        if self.role_specific_critic:
            return split_critic_prompt(first, second, task=self.task, direction=self.direction,
                                       raw_direction=self.raw_direction, evaluation_mode=self.evaluation_mode)
        return round_critic_prompt(first, second, task=self.task, direction=self.direction)

    def initial_critic_prompt(self, a, b):
        if self.role_specific_critic:
            return split_critic_prompt(a, b, task=self.task, direction=self.direction,
                                       raw_direction=self.raw_direction, initial=True,
                                       evaluation_mode=self.evaluation_mode)
        better = "Lower" if self.direction == "min" else "Higher"
        if a.fitness == b.fitness:
            return (f"{self.task}\nHere are the algorithms proposed by initial population.\n"
                    f"First code:\n{a.code}\nFirst objective score: {a.objective}\n"
                    f"Second code:\n{b.code}\nSecond objective score: {b.objective}\n"
                    "The objective scores are tied. Compare the designs without claiming either score "
                    "is better; do not assume a parent-child relationship or unsupported causality. "
                    "Give design hints in fewer than 20 words and actionable critique.\n"
                    "<ref>Reflection on candidate designs (max 30 words)</ref>\n"
                    "<ans>Specific critique and actionable suggestions (max 40 words)</ans>\n"
                    "Let's think step by step and be constructive in your analysis.")
        return (f"{self.task}\nHere are the algorithms proposed by initial population.\n"
                f"Better code:\n{a.code}\nBetter objective score: {a.objective}\n"
                f"Worse code:\n{b.code}\nWorse objective score: {b.objective}\n"
                f"Note: **A {better.lower()} objective score is better**.\n\n"
                "Act as a reflector and critical evaluator in the domain of optimization "
                "heuristics. First, compare the candidates and offer plausible explanations "
                "for the observed score difference, without assuming a parent-child relationship "
                "or claiming causality as fact. "
                "You respond with some hints for designing better heuristics, based on the two "
                "code and using less than 20 words. Then analyze the worse algorithm's "
                "limitations and provide specific, actionable improvements.\n\n"
                "Please provide your answer in the following format:\n"
                "<ref>Reflection on candidate designs (max 30 words)</ref>\n"
                "<ans>Specific critique and actionable suggestions (max 40 words)</ans>\n"
                "Let's think step by step and be constructive in your analysis.")

    def critic_views(self, response):
        if self.role_specific_critic:
            return response
        return {role: response for role in self.design_roles}

    def fusion_reflections(self, response):
        if not self.role_specific_critic:
            return [response["reflection"]]
        return [f"{self.role_details[role]['label']}: {response[role]['reflection']}"
                for role in self.design_roles]

    def round_fusion_prompt(self, proposals, better):
        first, second = proposals
        first_label, second_label = (self.role_details[role]["label"] for role in self.design_roles)
        wording = self.fusion_wording
        return (f"{self.task}\nHere are the current candidates for integration:\n"
                f"{first_label}'s algorithm description: {first.thought}\n{first_label}'s code: {first.code}\n"
                f"{first_label}'s objective Score: {first.objective}\n\n"
                f"{second_label}'s algorithm description: {second.thought}\n{second_label}'s code: {second.code}\n"
                f"{second_label}'s objective Score: {second.objective}\n\n"
                f"Act as a strategic integrator. Weigh {wording['round_priorities']}. "
                f"Remember A {better.lower()} objective score is better.\n"
                f"Propose the next step that balances {wording['balance']}, maximizing overall progress "
                "by reasoning through their complementary strengths.\n"
                "Do not claim measured runtime, quality preservation or causality beyond the supplied evidence.\n"
                f"{self.output}")

    def final_fusion_prompt(self, best, reflections):
        first, second = best
        wording = self.fusion_wording
        return (f"{self.task}\nHere are the BEST candidates for integration:\n"
                f"Best {wording['first_final_label']}'s code:{first.code}\nobjective Score: {first.objective}\n\n"
                f"Best {wording['second_final_label']}'s code:{second.code}\nobjective Score: {second.objective}\n\n"
                f"Reflection summary: {' | '.join(reflections)}\nAct as a strategic integrator. You must consider:\n"
                f"1. The {wording['first_final_priority']}\n"
                f"2. The {wording['second_final_priority']}\n"
                "3. Insights from the reflection summary\n"
                "Your task is to create a solution that combines the strongest elements from both algorithms "
                "while addressing insights from the reflection. Create an algorithm that balances "
                f"{wording['balance']} to maximize progress.\n"
                "Do not claim measured runtime, quality preservation or causality beyond the supplied evidence.\n"
                f"{self.output}")

    def long_reflection_prompt(self, role, reflections, changes):
        label = self.role_details[role]["label"]
        return (f"Below is your prior short-term reflection on designing heuristics for {self.task}\n"
                f"{' | '.join(reflections)}\n"
                f"Objective changes: {changes[role]}\n"
                f"Based on the relection with changing of the objective scores for {label} {self.role_focus(role)},\n"
                "write constructive and detailed hints for designing better heuristics, based on prior reflections "
                f"and focus on {label} and using less than 50 words. Treat causal explanations as hypotheses; "
                "a combined score does not establish separate runtime or quality changes. "
                "Please provide your answer in the following "
                "format: <ans>...</ans>. Let's think step by step.")

    def merge_reflections_prompt(self, lessons):
        first, second = self.design_roles
        return ("Merge these two reflections into one concise experience. Return plain text only.\n"
                f"{self.role_details[first]['label']}: {lessons[first]}\n"
                f"{self.role_details[second]['label']}: {lessons[second]}")

    def run(self):
        try:
            initial_prompt = f"{self.task}\nPropose a new heuristic without an existing heuristic.\n{self.output}"
            initial = self.batch([(f"init_{i+1:02d}", (), initial_prompt) for i in range(self.initial_count)])
            population = self.select(initial, self.n)
            self.event("initial_population", candidate_ids=[c.candidate_id for c in population])
            for generation in range(1, self.generations + 1):
                self.generation = generation
                self.backend.before_generation()
                eoh = []
                directions = {"E1": "Create a heuristic as different as possible from these parents.",
                              "E2": "Identify their shared idea and create a new structure based on it.",
                              "M1": "Modify this heuristic's structure to improve performance.",
                              "M2": "Adjust parameters while retaining this heuristic's overall logic."}
                specs = []
                for operator in ("E1", "E2", "M1", "M2"):
                    for i in range(self.n):
                        parents = self.eoh_parents(population, min(5, self.n) if operator in ("E1", "E2") else 1)
                        context = "\n".join(f"Algorithm description: {c.thought}\nCode:\n{c.code}\n" for c in parents)
                        specs.append((f"eoh_{operator}_{i+1:02d}", parents,
                                      f"{self.task}\n{context}{directions[operator]}\n{self.output}"))
                eoh = self.batch(specs)
                a, b = self.elite_pair(population)
                self.event("elite_pair", generation=self.generation, a=a.candidate_id, b=b.candidate_id)
                better = "Lower" if self.direction == "min" else "Higher"
                initial_critic = self.initial_critic_prompt(a, b)
                parser = parse_split_critic if self.role_specific_critic else probe.parse_critic
                critic = self.request("critic_initial", initial_critic, parser)
                views = self.critic_views(critic)
                reflections = self.fusion_reflections(critic)
                role_reflections = {role: [views[role]["reflection"]] for role in self.design_roles}
                first_role, second_role = self.design_roles
                all_roles = (*self.design_roles, self.integrator_role)
                starts = self.batch([(f"start_{role}", (b,), self.branch_prompt(role, b, views[role]))
                                     for role in self.design_roles])
                current = {role: result if result.valid else b for role, result in zip(self.design_roles, starts)}
                changes = {role: [{"from_objective": b.objective, "proposal_objective": result.objective,
                                   "to_objective": current[role].objective}] for role, result in zip(self.design_roles, starts)}
                for role, result in zip(self.design_roles, starts):
                    self.event("branch_transition", generation=self.generation, role=role, round=0, previous=b.candidate_id,
                               proposal=result.candidate_id, current=current[role].candidate_id, **changes[role][-1])
                formal = {role: [] for role in self.design_roles}
                last = []
                for turn in range(1, self.rounds + 1):
                    parser = parse_split_critic if self.role_specific_critic else parse_round_critic
                    critic = self.request(f"critic_{turn}", self.branch_critic_prompt(current[first_role], current[second_role]),
                                          parser)
                    views = self.critic_views(critic)
                    reflections.extend(self.fusion_reflections(critic))
                    for role in self.design_roles:
                        role_reflections[role].append(views[role]["reflection"])
                    previous = current.copy()
                    proposals = self.batch([(f"round_{turn}_{role}", (previous[role],),
                                             self.branch_prompt(role, previous[role], views[role]))
                                            for role in self.design_roles])
                    for role, result in zip(self.design_roles, proposals):
                        if result.valid:
                            formal[role].append(result)
                            current[role] = result
                        change = {"from_objective": previous[role].objective, "proposal_objective": result.objective,
                                  "to_objective": current[role].objective}
                        changes[role].append(change)
                        self.event("branch_transition", generation=self.generation, role=role, round=turn, previous=previous[role].candidate_id,
                                   proposal=result.candidate_id, current=current[role].candidate_id, **change)
                    fusion = None
                    if all(c.valid for c in proposals):
                        prompt = self.round_fusion_prompt(proposals, better)
                        fusion = self.batch([(f"fusion_{turn}", proposals, prompt)])[0]
                    else:
                        self.event("slot_skipped", generation=self.generation, stage=f"fusion_{turn}", reason="invalid_new_input")
                    last = [*proposals, fusion]
                elite_fusion = None
                if all(formal[role] for role in self.design_roles):
                    best = [self.select(formal[role], 1)[0] for role in self.design_roles]
                    prompt = self.final_fusion_prompt(best, reflections)
                    elite_fusion = self.batch([("fusion_final", best, prompt)])[0]
                else:
                    self.event("slot_skipped", generation=self.generation, stage="fusion_final", reason="missing_formal_branch")
                lessons = {}
                for role in self.design_roles:
                    prompt = self.long_reflection_prompt(role, role_reflections[role], changes)
                    lessons[role] = self.request(f"reflection_{role}", prompt, probe.parse_long_term)
                lessons[self.integrator_role] = self.request(f"reflection_{self.integrator_role}",
                                                             self.merge_reflections_prompt(lessons), probe.parse_merge)
                specs = []
                for elite_name, elite in (("A", a), ("B", b)):
                    for role in all_roles:
                        prompt = (f"{self.task}\n[History reflection as reference]:\n{' | '.join(self.history[role])}\n"
                                  f"[Reflection insights]:\n{lessons[role]}\n[Current best code]:\n{elite.code}\n"
                                  "Based on the history reflections and mainly based on reflection insights and "
                                  "the current best code, propose a mutation that addresses the key insights "
                                  f"and improves upon the previous solutions.\n{self.output}")
                        specs.append((f"mutation_{elite_name}_{role}", (elite,), prompt))
                mutations = self.batch(specs)
                pool = [*eoh, *last, elite_fusion, *mutations]
                selected = self.select(pool, self.n)
                self.event("selection", generation=self.generation, pool=[c.candidate_id for c in pool if c is not None],
                           selected=[{"candidate_id": c.candidate_id, "evaluation_origin": c.evaluation_origin}
                                     for c in selected])
                for role in all_roles:
                    self.history[role].append(lessons[role])
                    self.event("history_appended", role=role, generation=self.generation, reflection=lessons[role])
                population = selected
            self.event("finish", status="completed")
            return population
        except StopRun as exc:
            self.stop.set()
            cancelled = getattr(self.backend, "cancelled", None)
            if cancelled is not None:
                cancelled.set()
            self.event("finish", status="stopped", reason=str(exc))
            raise
        except (Exception, KeyboardInterrupt) as exc:
            self.stop.set()
            cancelled = getattr(self.backend, "cancelled", None)
            if cancelled is not None:
                cancelled.set()
            self.event("finish", status="stopped", reason=f"{type(exc).__name__}: {exc}")
            raise StopRun(f"运行意外停止：{type(exc).__name__}: {exc}") from exc
