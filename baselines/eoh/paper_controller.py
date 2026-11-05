"""论文五算子的 EoH 适配；不将异步新版或四算子版本冒充论文流程。"""

import re

import numpy as np

from scripts import p0_model_probe as probe
from scripts.p0_controller import Controller, StopRun


OPERATORS = ("E1", "E2", "M1", "M2", "M3")


def parse_proposal(contract, response):
    """读取 EoH 的花括号思想和完整代码；无效输出不额外重试。"""
    if not isinstance(response, str):
        raise probe.ProbeError("EoH 模型输出不是字符串。")
    match = re.match(r"\s*\{([^{}]+)\}\s*", response, flags=re.DOTALL)
    if not match or not match.group(1).strip():
        raise probe.ProbeError("EoH 输出须先包含花括号思想。")
    thought = match.group(1).strip()
    code = response[match.end():].strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n(.*?)\s*```", code, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        code = fenced.group(1).strip()
    return contract.parse_proposal(f"<thought>{thought}</thought><code>{code}</code>")


def make_prompt(contract, operator, parents):
    """保留来源的思想／代码输入与五类策略，任务接口采用本项目合同。"""
    if operator == "I1":
        if parents:
            raise ValueError("EoH 初始化不接受父代。")
        context = ""
        instruction = "Create a new heuristic."
    elif operator in ("E1", "E2"):
        if len(parents) != 5:
            raise ValueError("E1/E2 需要五次父代抽样。")
        context = "\n".join(f"No.{i} algorithm and code:\n{item.thought}\n{item.code}"
                            for i, item in enumerate(parents, 1))
        instruction = ("Create a new algorithm with a totally different form from these parents."
                       if operator == "E1" else
                       "Identify the parents' common backbone, then create a new algorithm motivated by it.")
    elif operator in ("M1", "M2", "M3"):
        if len(parents) != 1:
            raise ValueError("M1/M2/M3 需要一个父代。")
        item = parents[0]
        context = f"Algorithm description: {item.thought}\nCode:\n{item.code}"
        instruction = {
            "M1": "Modify this heuristic to produce a different algorithm with better performance.",
            "M2": "Identify the main parameters and produce a version with different parameter settings.",
            "M3": "Identify the main components and redundant parts, then simplify the implementation.",
        }[operator]
    else:
        raise ValueError(f"未知 EoH 算子：{operator}")
    return (f"{contract.task_prompt()}\n{context}\n{instruction}\n"
            "First describe the algorithm in one sentence inside braces. Then provide the complete "
            f"executable Python module with import numpy as np and {contract.signature}\n"
            "Do not use tools or read files. Do not add other explanations.")


class EoHController(Controller):
    """论文同步代际流程；保留 2024 官方代码的排名抽样与五位小数去重。"""

    def __init__(self, backend, log, contract, *, population_size=10, generations=1, seed=20261103):
        super().__init__(backend, log, population_size=population_size, initial_count=population_size,
                         rounds=3, seed=seed, task_contract=contract, generations=generations)
        self.contract = contract
        self.parse_proposal = lambda response: parse_proposal(contract, response)

    def parents(self, population, count):
        weights = [1 / (rank + 1 + len(population)) for rank in range(len(population))]
        return tuple(self.random.choices(population, weights=weights, k=count))

    def select_eoh(self, candidates):
        """来源按最小化、五位小数目标去重；Fitness 已统一为越大越好。"""
        unique = {}
        for candidate in candidates:
            if candidate is not None and candidate.valid:
                score = float(np.round(-candidate.fitness, 5))
                unique.setdefault(score, candidate)
        selected = [candidate for _score, candidate in sorted(unique.items(), key=lambda item: item[0])[:self.n]]
        if len(selected) < self.n:
            raise StopRun(f"EoH 有效不同目标仅 {len(selected)} 个，少于所需 {self.n} 个。")
        return selected

    def run(self):
        try:
            prompt = make_prompt(self.contract, "I1", ())
            initial = self.batch([(f"init_{i+1:02d}", (), prompt) for i in range(self.n)])
            population = self.select_eoh(initial)
            self.event("initial_population", candidate_ids=[item.candidate_id for item in population])
            for generation in range(1, self.generations + 1):
                self.generation = generation
                self.backend.before_generation()
                specs = []
                for operator in OPERATORS:
                    for i in range(self.n):
                        parents = self.parents(population, 5 if operator in ("E1", "E2") else 1)
                        specs.append((f"eoh_{operator}_{i+1:02d}", parents,
                                      make_prompt(self.contract, operator, parents)))
                offspring = self.batch(specs)
                pool = [*population, *offspring]
                population = self.select_eoh(pool)
                self.event("selection", generation=generation,
                           pool=[item.candidate_id for item in pool],
                           selected=[{"candidate_id": item.candidate_id,
                                      "search_objective": float(np.round(-item.fitness, 5))}
                                     for item in population])
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
            raise StopRun(f"EoH 运行意外停止：{type(exc).__name__}: {exc}") from exc
