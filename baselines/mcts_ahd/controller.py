"""MCTS-AHD 固定来源的搜索树与生成流程适配。"""

from dataclasses import dataclass, field
import math
import re
import time

import numpy as np

from baselines.reevo.controller import plain_response
from scripts import p0_model_probe as probe
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope


@dataclass
class Node:
    candidate: Candidate | None
    parent: "Node | None" = None
    depth: int = 0
    visits: int = 0
    q: float = 0.0
    children: list = field(default_factory=list)
    subtree: list = field(default_factory=list)


def search_objective(candidate):
    """来源按最小化且保留五位小数；P1 Fitness 统一为越大越好。"""
    return float(np.round(-candidate.fitness, 5))


def parse_proposal(contract, response):
    """来源允许 E2/S1 在花括号思想之前先列出共同思路。"""
    if not isinstance(response, str):
        raise probe.ProbeError("MCTS-AHD 输出不是字符串。")
    match = re.search(r"\{([^{}]+)\}", response, flags=re.DOTALL)
    if not match or not match.group(1).strip():
        raise probe.ProbeError("MCTS-AHD 输出缺少花括号思想。")
    code = response[match.end():].strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n(.*?)\s*```", code, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        code = fenced.group(1).strip()
    return contract.parse_proposal(f"<thought>{match.group(1).strip()}</thought><code>{code}</code>")


def prompt_for(contract, operator, parents):
    descriptions = "\n\n".join(
        f"No.{i} algorithm's description, code and objective value are:\n{item.thought}\n"
        f"{item.code}\nObjective value: {search_objective(item)}"
        for i, item in enumerate(parents, 1))
    instructions = {
        "i1": "Create a new heuristic algorithm.",
        "e1": "Create a new algorithm with a totally different form from the given algorithms, with a relatively low objective value.",
        "e2": "Create an algorithm similar to the last algorithm and inspired by the first, aiming for a lower objective than both.",
        "m1": "Create a modified algorithm with a different form, novel mechanisms or equations.",
        "m2": "Identify the main parameters and create an algorithm with different parameter settings.",
        "s1": "Use the ideas in all algorithms on this search path to create a better algorithm.",
    }
    if operator not in instructions:
        raise ValueError("未知 MCTS-AHD 算子。")
    lead = ("First list useful common ideas. Then describe the new idea in one sentence inside braces. "
            if operator in ("e2", "s1") else "First describe the idea in one sentence inside braces. ")
    return (f"{contract.task_prompt()}\n{descriptions}\n{instructions[operator]}\n"
            f"Lower objective is better. {lead}"
            "Then provide the complete executable Python module including import numpy as np and "
            f"{contract.signature}\nDo not use tools or read files. Do not add other explanations.")


def refine_prompt(contract, candidate):
    return (f"{contract.task_prompt()}\nFollowing is the design idea and code of a heuristic.\n"
            f"Design Idea:\n{candidate.thought}\nCode:\n{candidate.code}\n"
            "Re-describe the most critical design ideas of the code in fewer than three sentences.")


class MCTSAHDController(Controller):
    """树节点、渐进扩展、UCT 衰减、路径提示与独立思想重述。"""

    role_specific_critic = False

    def __init__(self, backend, log, contract, *, initial_size=None, population_size=10,
                 max_evaluations=50, seed=20261103):
        initial_size = (10 if contract.black_box else 4) if initial_size is None else initial_size
        if (type(initial_size) is not int or initial_size < 2 or type(max_evaluations) is not int
                or max_evaluations < initial_size or type(seed) is not int or seed < 0):
            raise ValueError("MCTS-AHD 初始化数量、评价上限或随机种子无效。")
        super().__init__(backend, log, population_size=population_size, initial_count=initial_size,
                         rounds=3, seed=seed, task_contract=contract, generations=1)
        self.contract, self.max_evaluations = contract, max_evaluations
        self.attempts = 0
        self.root = Node(None)
        self.q_min, self.q_max = 0.0, -10000.0
        self.rank_list = []

    def register(self, stage, parents):
        with self._lock:
            sequence = len(self.candidates) + 1
            candidate = Candidate(f"c{sequence:04d}", sequence, stage, parents)
            self.candidates.append(candidate)
        self.event("candidate_registered", candidate_id=candidate.candidate_id, sequence=sequence,
                   stage=stage, source_role="MCTS-AHD", operator=stage.split("_", 1)[0],
                   generation=None, round=None, parents=list(parents))
        return candidate

    def create_candidate(self, operator, parents):
        if self.attempts >= self.max_evaluations:
            return None
        before_step = getattr(self.backend, "before_search_step", None)
        if before_step is not None:
            before_step()
        self.attempts += 1
        stage = f"{operator}_{self.attempts:04d}"
        candidate = self.register(stage, tuple(parent.candidate_id for parent in parents))
        self.backend.prepare_proposal(candidate)
        try:
            prompt = prompt_for(self.contract, operator, parents)
            response = self.backend.model(stage, prompt)
            self._generated += 1
            candidate.generation_order = self._generated
            self.event("proposal_response", candidate_id=candidate.candidate_id, stage=stage,
                       generation_order=candidate.generation_order, prompt=prompt, response=response)
            try:
                parsed = parse_proposal(self.contract, response)
                candidate.thought, candidate.code = parsed["thought"], parsed["code"]
                check_modification_scope(candidate.code)
            except (probe.ProbeError, ValueError, SyntaxError) as exc:
                candidate.error = f"invalid_proposal: {exc}"
                self.event("candidate_invalid", candidate_id=candidate.candidate_id, error=candidate.error)
                raise StopRun(f"MCTS-AHD 候选无效，未补提案：{exc}") from exc
            candidate.thought = self.request(f"post_thought_{self.attempts:04d}",
                                             refine_prompt(self.contract, candidate), plain_response)
            result = self.backend.evaluate(candidate)
            self.event("evaluation", candidate_id=candidate.candidate_id, result=result)
            if result.get("status") != "passed":
                candidate.error = str(result.get("error") or result.get("status"))
                raise StopRun(f"MCTS-AHD 候选评价失败，未补提案：{candidate.error}")
            objective, fitness = result.get("average"), result.get("fitness")
            if (isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective)
                    or isinstance(fitness, bool) or not isinstance(fitness, (int, float)) or not math.isfinite(fitness)
                    or fitness != (-objective if self.direction == "min" else objective)):
                raise StopRun("MCTS-AHD 评价目标或 Fitness 不可信。")
            candidate.objective, candidate.fitness = float(objective), float(fitness)
            candidate.completed_order = result.get("completed_monotonic_ns", time.monotonic_ns())
            candidate.evaluation_origin = candidate.candidate_id
            return candidate
        finally:
            self.backend.release_proposal(candidate)

    def add_node(self, candidate, parent):
        node = Node(candidate, parent=parent, depth=parent.depth + 1, visits=1, q=-search_objective(candidate))
        parent.children.append(node)
        if parent is self.root:
            node.subtree.append(node)
        if node.q not in self.rank_list:
            self.rank_list.append(node.q)
            self.rank_list.sort()
        self.q_min, self.q_max = min(self.q_min, node.q), max(self.q_max, node.q)
        ancestor = parent
        while ancestor is not None:
            ancestor.q = max(child.q for child in ancestor.children)
            ancestor.visits += 1
            if ancestor.parent is self.root:
                ancestor.subtree.append(node)
            ancestor = ancestor.parent
        self.event("tree_node_added", candidate_id=candidate.candidate_id,
                   parent=parent.candidate.candidate_id if parent.candidate else "root", depth=node.depth,
                   q=node.q, q_min=self.q_min, q_max=self.q_max)
        return node

    def manage(self, candidates):
        unique = {}
        for item in candidates:
            if item.valid:
                unique.setdefault(search_objective(item), item)
        return [item for _score, item in sorted(unique.items())[:self.n]]

    def e1_parents(self, candidates):
        count = min(self.random.randint(2, 5), len(candidates))
        if count < 1:
            raise StopRun("MCTS-AHD 缺少可供 E1 使用的父代。")
        return tuple(self.random.choices(candidates, k=count))

    def path_parents(self, node):
        path = []
        while node.candidate is not None:
            path.append(node.candidate)
            node = node.parent
        unique = {}
        for item in path:
            unique.setdefault(item.thought, item)
        return tuple(sorted(unique.values(), key=search_objective, reverse=True))

    def expand(self, node, population, operator):
        if self.attempts >= self.max_evaluations:
            return population
        if operator == "e1":
            candidates = [self.random.choice(child.subtree).candidate for child in self.root.children]
            parents = self.e1_parents(candidates)
        elif operator == "e2":
            others = [item for item in population if item.candidate_id != node.candidate.candidate_id]
            if not others:
                raise StopRun("MCTS-AHD E2 缺少另一父代。")
            weights = [1 / (rank + 1 + len(others)) for rank in range(len(others))]
            parents = (self.random.choices(others, weights=weights, k=1)[0], node.candidate)
        elif operator in ("m1", "m2"):
            parents = (node.candidate,)
        elif operator == "s1":
            parents = self.path_parents(node)
            if len(parents) < 2:
                self.event("path_expansion_skipped", candidate_id=node.candidate.candidate_id, path_size=len(parents))
                return population
        else:
            raise ValueError("未知 MCTS-AHD 扩展算子。")
        offspring = self.create_candidate(operator, parents)
        if offspring is None:
            return population
        if operator == "e1" and any(search_objective(child.candidate) == search_objective(offspring)
                                    for child in self.root.children):
            self.event("root_expansion_discarded", candidate_id=offspring.candidate_id, reason="duplicate_objective")
            return population
        if operator != "e1" and any(item.code == offspring.code for item in population):
            raise StopRun("MCTS-AHD 新候选与当前种群代码重复；未补提案。")
        population = self.manage([*population, offspring])
        self.add_node(offspring, node)
        return population

    def uct(self, node):
        if self.q_max == self.q_min:
            raise StopRun("MCTS-AHD 的 Q 范围为零，无法计算 UCT。")
        remaining = max(1 - self.attempts / self.max_evaluations, 0)
        return ((node.q - self.q_min) / (self.q_max - self.q_min)
                + 0.1 * remaining * math.sqrt(math.log(node.parent.visits + 1) / node.visits))

    def run(self):
        try:
            initial = []
            for i in range(self.initial_count):
                operator = "i1" if i == 0 else "e1"
                parents = () if i == 0 else self.e1_parents(initial)
                candidate = self.create_candidate(operator, parents)
                if candidate is None or any(search_objective(item) == search_objective(candidate) for item in initial):
                    raise StopRun("MCTS-AHD 初始节点不足或目标重复；未补提案。")
                initial.append(candidate)
                self.add_node(candidate, self.root)
            population = self.manage(initial)
            self.event("initial_population", candidate_ids=[item.candidate_id for item in population])
            while self.attempts < self.max_evaluations:
                node = self.root
                while node.children and node.depth < 10:
                    scores = [self.uct(child) for child in node.children]
                    selected = scores.index(max(scores))
                    if int(node.visits ** 0.5) > len(node.children):
                        population = self.expand(node, population, "e1" if node is self.root else "e2")
                    node = node.children[selected]
                    self.event("tree_descent", candidate_id=node.candidate.candidate_id,
                               depth=node.depth, uct_scores=scores)
                for operator, count in (("e2", 1), ("m1", 2), ("m2", 2), ("s1", 1)):
                    for _ in range(count):
                        population = self.expand(node, population, operator)
                self.event("search_population", attempts=self.attempts,
                           candidate_ids=[item.candidate_id for item in population])
            if not population:
                raise StopRun("MCTS-AHD 没有有效候选。")
            self.event("finish", status="completed", attempts=self.attempts, best=population[0].candidate_id)
            return population[0]
        except StopRun as exc:
            self.stop.set()
            cancelled = getattr(self.backend, "cancelled", None)
            if cancelled is not None:
                cancelled.set()
            self.event("finish", status="stopped", attempts=self.attempts, reason=str(exc))
            raise
        except (Exception, KeyboardInterrupt) as exc:
            self.stop.set()
            cancelled = getattr(self.backend, "cancelled", None)
            if cancelled is not None:
                cancelled.set()
            self.event("finish", status="stopped", attempts=self.attempts, reason=f"{type(exc).__name__}: {exc}")
            raise StopRun(f"MCTS-AHD 运行意外停止：{type(exc).__name__}: {exc}") from exc
