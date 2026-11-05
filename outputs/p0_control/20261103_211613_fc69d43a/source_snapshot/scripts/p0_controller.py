"""P0 一代控制：角色输入、候选来源、失败回退和固定候选池。"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import random
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
    completed_order: int | None = None
    evaluation_origin: str | None = None
    error: str | None = None

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


def candidate_text(candidate: Candidate, *, thought: bool = True) -> str:
    desc = f"Algorithm description: {candidate.thought}\n" if thought else ""
    return f"{desc}Code:\n{candidate.code}\nObjective score: {candidate.objective}\n"


def role_prompt(role: str, candidate: Candidate, critic: dict[str, str]) -> str:
    focus = "improving the objective value" if role == "Q" else "reducing computational cost"
    style = ("Use creative, future-oriented thinking to introduce novel directions combined with the feedback."
             if role == "Q" else
             "Use conservative, efficiency-driven thinking to enhance performance incrementally.")
    return (f"{TASK}\n{candidate_text(candidate)}Critic's feedback: {critic['advice']}\n"
            f"Reflection: {critic['reflection']}\nBased on the feedback and reflection, propose a new "
            f"algorithm that improves the previous one, focusing on {focus}. {style}\n{OUTPUT}")


def round_critic_prompt(q: Candidate, e: Candidate) -> str:
    if e.fitness > q.fitness:
        instruction = ("The current E candidate has a better objective than Q. Identify successful changes "
                       "and recommend how to build on them.")
        compared = f"Better algorithm (E):\n{candidate_text(e)}Worse algorithm (Q):\n{candidate_text(q)}"
    elif e.fitness < q.fitness:
        instruction = ("The current E candidate has a worse objective than Q. Identify counterproductive "
                       "choices and suggest how to correct the direction.")
        compared = f"Better algorithm (Q):\n{candidate_text(q)}Worse algorithm (E):\n{candidate_text(e)}"
    else:
        instruction = "The objectives are tied. Compare the designs without claiming either score is better."
        compared = f"Q algorithm:\n{candidate_text(q)}E algorithm:\n{candidate_text(e)}"
    return (f"{TASK}\nCompare these evaluated algorithms; a lower objective is better.\n"
            f"{compared}{instruction} Reflect on the differences and give actionable improvements. "
            "Return exactly <ref>reflection</ref><ans>advice, at most 40 words</ans>.")


def parse_round_critic(message: str) -> dict[str, str]:
    reflection, advice = probe.tagged(message, ("ref", "ans"))
    if probe.english_words(advice) > 40:
        raise probe.ProbeError("逐轮 Critic 建议超过 40 词。")
    return {"reflection": reflection, "advice": advice}


class Controller:
    """backend 提供 model、evaluate、before_generation；log 提供 event。"""

    def __init__(self, backend, log, *, population_size=10, initial_count=30, rounds=3, seed=20261103):
        self.backend, self.log = backend, log
        self.n, self.initial_count, self.rounds = population_size, initial_count, rounds
        self.random = random.Random(seed)
        self.candidates: list[Candidate] = []
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.history: dict[str, list[str]] = {role: [] for role in ("Q", "E", "I")}

    def event(self, kind: str, **fields: Any) -> None:
        self.log.event(kind, **fields)

    def request(self, stage: str, prompt: str, parser):
        if self.stop.is_set():
            raise StopRun("已有并发任务停止本次运行。")
        try:
            response = self.backend.model(stage, prompt)
            self.event("model_response", stage=stage, prompt=prompt, response=response)
            return parser(response)
        except probe.ProbeError as exc:
            self.event("invalid_role_response", stage=stage, error=str(exc))
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
        parts = stage.split("_")
        if parts[0] == "init":
            role, operator, generation, round_number = "initializer", "initialization", 0, None
        elif parts[0] == "eoh":
            role, operator, generation, round_number = "EoH", parts[1], 1, None
        elif parts[0] == "start":
            role, operator, generation, round_number = parts[1], "startup", 1, 0
        elif parts[0] == "round":
            role, operator, generation, round_number = parts[2], "continuation", 1, int(parts[1])
        elif parts[0] == "fusion":
            role, operator, generation = "I", "elite_fusion" if parts[1] == "final" else "round_fusion", 1
            round_number = None if parts[1] == "final" else int(parts[1])
        else:
            role, operator, generation, round_number = parts[2], "elite_mutation", 1, None
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
            self.event("proposal_response", candidate_id=candidate.candidate_id, stage=candidate.stage,
                       prompt=prompt, response=response)
            try:
                parsed = probe.parse_proposal(response)
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
            import math
            if not math.isfinite(objective) or not math.isfinite(fitness) or fitness != -objective:
                raise StopRun("评价器返回的目标或 Fitness 不可信。")
            with self._lock:
                candidate.completed_order = result.get("completed_monotonic_ns", time.monotonic_ns())
            candidate.objective, candidate.fitness = float(objective), float(fitness)
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
        earliest = {}
        for item in sorted(all_valid, key=lambda c: c.completed_order):
            earliest.setdefault(code_hash(item.code), item)
        unique = {}
        for item in sorted(valid, key=lambda c: c.sequence):
            key = code_hash(item.code)
            origin = earliest[key]
            unique.setdefault(key, replace(item, objective=origin.objective, fitness=origin.fitness,
                                           evaluation_origin=origin.candidate_id))
        selected = sorted(unique.values(), key=lambda c: (-c.fitness, c.sequence))
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

    def run(self):
        try:
            initial_prompt = f"{TASK}\nPropose a new heuristic without an existing heuristic.\n{OUTPUT}"
            initial = self.batch([(f"init_{i+1:02d}", (), initial_prompt) for i in range(self.initial_count)])
            population = self.select(initial, self.n)
            self.event("initial_population", candidate_ids=[c.candidate_id for c in population])
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
                                  f"{TASK}\n{context}{directions[operator]}\n{OUTPUT}"))
            eoh = self.batch(specs)
            a, b = self.elite_pair(population)
            self.event("elite_pair", a=a.candidate_id, b=b.candidate_id)
            initial_critic = (f"{TASK}\nBetter code:\n{a.code}\nBetter objective score: {a.objective}\n"
                              f"Worse code:\n{b.code}\nWorse objective score: {b.objective}\n"
                              "Lower objective is better. Reflect on why the better code performs better; "
                              "give design hints in fewer than 20 words and actionable critique. "
                              "Return exactly <ref>reflection</ref><ans>advice, at most 40 words</ans>.")
            critic = self.request("critic_initial", initial_critic, probe.parse_critic)
            reflections = [critic["reflection"]]
            starts = self.batch([(f"start_{role}", (b,), role_prompt(role, b, critic)) for role in ("Q", "E")])
            current = {role: result if result.valid else b for role, result in zip(("Q", "E"), starts)}
            changes = {role: [{"from_objective": b.objective, "proposal_objective": result.objective,
                               "to_objective": current[role].objective}] for role, result in zip(("Q", "E"), starts)}
            for role, result in zip(("Q", "E"), starts):
                self.event("branch_transition", role=role, round=0, previous=b.candidate_id,
                           proposal=result.candidate_id, current=current[role].candidate_id, **changes[role][-1])
            formal = {"Q": [], "E": []}
            last = []
            for turn in range(1, self.rounds + 1):
                critic = self.request(f"critic_{turn}", round_critic_prompt(current["Q"], current["E"]),
                                      parse_round_critic)
                reflections.append(critic["reflection"])
                previous = current.copy()
                proposals = self.batch([(f"round_{turn}_{role}", (previous[role],),
                                         role_prompt(role, previous[role], critic)) for role in ("Q", "E")])
                for role, result in zip(("Q", "E"), proposals):
                    if result.valid:
                        formal[role].append(result)
                        current[role] = result
                    change = {"from_objective": previous[role].objective, "proposal_objective": result.objective,
                              "to_objective": current[role].objective}
                    changes[role].append(change)
                    self.event("branch_transition", role=role, round=turn, previous=previous[role].candidate_id,
                               proposal=result.candidate_id, current=current[role].candidate_id, **change)
                fusion = None
                if all(c.valid for c in proposals):
                    prompt = (f"{TASK}\nQ candidate:\n{candidate_text(proposals[0])}"
                              f"E candidate:\n{candidate_text(proposals[1])}"
                              "Act as a strategic integrator. Weigh the long-term innovation of Q and the "
                              "short-term refinement of E. A lower objective is better. Propose the next "
                              "step that balances exploration and exploitation by reasoning through "
                              f"their complementary strengths.\n{OUTPUT}")
                    fusion = self.batch([(f"fusion_{turn}", proposals, prompt)])[0]
                else:
                    self.event("slot_skipped", stage=f"fusion_{turn}", reason="invalid_new_input")
                last = [*proposals, fusion]
            elite_fusion = None
            if formal["Q"] and formal["E"]:
                best = [self.select(formal[role], 1)[0] for role in ("Q", "E")]
                prompt = (f"{TASK}\nBest Q:\n{candidate_text(best[0], thought=False)}"
                          f"Best E:\n{candidate_text(best[1], thought=False)}"
                          f"Reflection summary: {' | '.join(reflections)}\nAct as a strategic integrator. "
                          "Consider the long-term innovation potential of the best Q, the short-term "
                          "refinement capabilities of the best E, and the reflection insights. Combine "
                          "their strongest elements and address those insights, balancing exploration "
                          f"and exploitation.\n{OUTPUT}")
                elite_fusion = self.batch([("fusion_final", best, prompt)])[0]
            else:
                self.event("slot_skipped", stage="fusion_final", reason="missing_formal_branch")
            lessons = {}
            for role in ("Q", "E"):
                focus = "improving the objective value" if role == "Q" else "reducing computational cost"
                prompt = (f"{TASK}\nRole: {role}, focus on {focus}.\n"
                          f"Short-term reflections: {' | '.join(reflections)}\n"
                          f"Objective changes: {changes[role]}\n"
                          "Give constructive hints in fewer than 50 words. Return exactly <ans>...</ans>.")
                lessons[role] = self.request(f"reflection_{role}", prompt, probe.parse_long_term)
            lessons["I"] = self.request("reflection_I", f"Merge these two reflections into one concise experience. "
                                        f"Return plain text only.\nQ: {lessons['Q']}\nE: {lessons['E']}",
                                        probe.parse_merge)
            specs = []
            for elite_name, elite in (("A", a), ("B", b)):
                for role in ("Q", "E", "I"):
                    prompt = (f"{TASK}\nHistory reflection: {' | '.join(self.history[role])}\n"
                              f"Reflection insights: {lessons[role]}\nCurrent elite code:\n{elite.code}\n"
                              f"Propose a mutation based on these insights.\n{OUTPUT}")
                    specs.append((f"mutation_{elite_name}_{role}", (elite,), prompt))
            mutations = self.batch(specs)
            pool = [*eoh, *last, elite_fusion, *mutations]
            selected = self.select(pool, self.n)
            self.event("selection", pool=[c.candidate_id for c in pool if c is not None],
                       selected=[{"candidate_id": c.candidate_id, "evaluation_origin": c.evaluation_origin}
                                 for c in selected])
            for role in ("Q", "E", "I"):
                self.history[role].append(lessons[role])
                self.event("history_appended", role=role, generation=1, reflection=lessons[role])
            self.event("finish", status="completed")
            return selected
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
