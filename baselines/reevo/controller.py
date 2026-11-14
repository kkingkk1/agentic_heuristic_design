"""ReEvo 固定来源的反思、交叉和精英变异适配。"""

from concurrent.futures import ThreadPoolExecutor
import math
import re
import time
from pathlib import Path

import numpy as np

from scripts import p0_model_probe as probe
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope


UPSTREAM = Path(__file__).resolve().parent / "upstream" / "prompts"


def text(path):
    return path.read_text(encoding="utf-8").strip()


def filtered_code(code):
    """对应固定提交的 filter_code。"""
    result = []
    for line in code.split("\n"):
        if line.startswith(("def", "import", "from")):
            continue
        result.append(line)
        if line.startswith("return"):
            break
    return "\n".join(result)


def source_files(contract):
    stem = contract.name + ("_black_box" if contract.black_box else "")
    directory = UPSTREAM / stem
    common = UPSTREAM / "common"
    names = ("system_generator", "user_generator", "seed", "system_reflector", "user_reflector_st",
             "user_reflector_st_black_box", "user_reflector_lt", "crossover", "mutation")
    files = {name: common / f"{name}.txt" for name in names}
    files["seed_func"] = directory / "seed_func.txt"
    knowledge = directory / "external_knowledge.txt"
    if knowledge.is_file():
        files["external_knowledge"] = knowledge
    return files


def parse_generated(contract, response):
    """接受来源的 Python 代码块；保留完整模块并执行项目接口检查。"""
    if not isinstance(response, str):
        raise probe.ProbeError("ReEvo 模型输出不是字符串。")
    fenced = re.search(r"```python\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    code = fenced.group(1).strip() if fenced else response.strip()
    if not re.search(r"(?m)^\s*import\s+numpy\s+as\s+np\s*$", code):
        code = "import numpy as np\n" + code
    return contract.parse_proposal(f"<thought>ReEvo proposal</thought><code>{code}</code>")


def seed_code(contract, raw):
    if raw.count("heuristics_v1") != 1:
        raise ValueError("来源种子函数身份不明确。")
    code = raw.replace("heuristics_v1", "heuristics_v2")
    if contract.name == "op_aco" and contract.black_box:
        if code.count("edge_constraint") != 1:
            raise ValueError("OP 黑盒种子参数不符合已核实来源。")
        code = code.replace("edge_constraint", "node_constraint")
    return parse_generated(contract, code)["code"]


def plain_response(response):
    if not isinstance(response, str) or not response.strip():
        raise probe.ProbeError("ReEvo 反思输出为空。")
    return response.strip()


class ReEvoController(Controller):
    """保留来源种子、随机配对、短／长期反思、交叉与精英变异。"""

    role_specific_critic = False

    def __init__(self, backend, log, contract, *, population_size=10, initial_count=30,
                 mutation_count=5, generations=1, seed=20261103):
        super().__init__(backend, log, population_size=population_size, initial_count=initial_count,
                         rounds=3, seed=seed, task_contract=contract, generations=generations)
        if type(mutation_count) is not int or mutation_count < 1:
            raise ValueError("ReEvo 变异数量必须为正整数。")
        self.contract, self.mutation_count = contract, mutation_count
        self.parse_proposal = lambda response: parse_generated(contract, response)
        self.random_state = np.random.RandomState(seed)
        self.files = source_files(contract)
        self.prompts = {name: text(path) for name, path in self.files.items()}
        self.external_knowledge = self.prompts.get("external_knowledge", "")
        self.long_reflection = self.external_knowledge
        self.elite = self.seed = None

    def register(self, stage, parents):
        with self._lock:
            sequence = len(self.candidates) + 1
            candidate = Candidate(f"c{sequence:04d}", sequence, stage, parents)
            self.candidates.append(candidate)
        operator = "seed" if stage == "seed" else stage.split("_", 1)[0]
        self.event("candidate_registered", candidate_id=candidate.candidate_id, sequence=sequence,
                   stage=stage, source_role="ReEvo", operator=operator,
                   generation=self.generation, round=None, parents=list(parents))
        return candidate

    def evaluate_seed(self):
        candidate = self.register("seed", ())
        candidate.code = seed_code(self.contract, self.prompts["seed_func"])
        candidate.thought = "ReEvo reference seed"
        check_modification_scope(candidate.code)
        self.backend.prepare_proposal(candidate)
        try:
            result = self.backend.evaluate(candidate)
        finally:
            self.backend.release_proposal(candidate)
        self.event("evaluation", candidate_id=candidate.candidate_id, result=result)
        if result.get("status") != "passed":
            raise StopRun(f"ReEvo 种子评价失败：{result.get('error') or result.get('status')}")
        objective, fitness = result.get("average"), result.get("fitness")
        if (not isinstance(objective, (int, float)) or isinstance(objective, bool)
                or not isinstance(fitness, (int, float)) or isinstance(fitness, bool)
                or not math.isfinite(objective) or not math.isfinite(fitness)):
            raise StopRun("ReEvo 种子目标或 Fitness 无效。")
        expected = -objective if self.direction == "min" else objective
        if fitness != expected:
            raise StopRun("ReEvo 种子目标或 Fitness 无效。")
        candidate.objective, candidate.fitness = float(objective), float(fitness)
        candidate.generation_order, candidate.completed_order = 0, time.monotonic_ns()
        candidate.evaluation_origin = candidate.candidate_id
        self.seed = self.elite = candidate
        return candidate

    def update_elite(self, population):
        for candidate in population:
            if candidate.valid and (self.elite is None or candidate.fitness > self.elite.fitness):
                self.elite = candidate

    def parent_pairs(self, population):
        pool = list(population)
        if self.elite is not None and all(item.candidate_id != self.elite.candidate_id for item in pool):
            pool.insert(0, self.elite)
        pool = [item for item in pool if item.valid and (not self.contract.black_box or item.fitness > self.seed.fitness)]
        if len(pool) < 2:
            raise StopRun("ReEvo 有效父代不足；黑盒条件还要求优于种子。")
        pairs, trials = [], 0
        while len(pairs) < self.n:
            trials += 1
            if trials > 1000:
                raise StopRun("ReEvo 无法抽到目标不同的父代。")
            indices = self.random_state.choice(len(pool), size=2, replace=False)
            first, second = pool[int(indices[0])], pool[int(indices[1])]
            if first.objective != second.objective:
                worse, better = (first, second) if first.fitness < second.fitness else (second, first)
                pairs.append((worse, better))
        return pairs

    def request_batch(self, specs):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.request, stage, prompt, plain_response) for stage, prompt in specs]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    self.stop.set()
                    raise StopRun(f"ReEvo 反思阶段停止：{exc}") from exc
            return results

    def initial_prompt(self):
        user_generator = self.prompts["user_generator"].format(
            func_name="heuristics", problem_desc=self.contract.description,
            func_desc=self.contract.function_description)
        seed_prompt = self.prompts["seed"].format(seed_func=self.prompts["seed_func"], func_name="heuristics")
        output_contract = f"Use the exact interface {self.contract.signature} and include import numpy as np."
        return f"{self.prompts['system_generator']}\n{user_generator}\n{seed_prompt}\n{self.long_reflection}\n{output_contract}"

    def run(self):
        try:
            self.evaluate_seed()
            system_generator = self.prompts["system_generator"]
            system_reflector = self.prompts["system_reflector"]
            user_generator = self.prompts["user_generator"].format(
                func_name="heuristics", problem_desc=self.contract.description,
                func_desc=self.contract.function_description)
            output_contract = f"Use the exact interface {self.contract.signature} and include import numpy as np."
            initial_prompt = self.initial_prompt()
            initial = self.batch([(f"init_{i+1:02d}", (), initial_prompt) for i in range(self.initial_count)])
            population = initial
            self.update_elite(population)
            self.event("initial_population", candidate_ids=[item.candidate_id for item in population],
                       seed_candidate_id=self.seed.candidate_id)
            for generation in range(1, self.generations + 1):
                self.generation = generation
                self.backend.before_generation()
                pairs = self.parent_pairs(population)
                short_key = "user_reflector_st_black_box" if self.contract.black_box else "user_reflector_st"
                short_specs = []
                for index, (worse, better) in enumerate(pairs):
                    user = self.prompts[short_key].format(func_name="heuristics", func_desc=self.contract.function_description,
                                                          problem_desc=self.contract.description,
                                                          worse_code=filtered_code(worse.code), better_code=filtered_code(better.code))
                    short_specs.append((f"short_reflection_{index+1:02d}", f"{system_reflector}\n{user}"))
                reflections = self.request_batch(short_specs)
                signature0 = self.contract.signature.replace("heuristics_v2", "heuristics_v0")
                signature1 = self.contract.signature.replace("heuristics_v2", "heuristics_v1")
                crossover_specs = []
                for index, ((worse, better), reflection) in enumerate(zip(pairs, reflections)):
                    user = self.prompts["crossover"].format(
                        user_generator=user_generator, func_signature0=signature0, func_signature1=signature1,
                        worse_code=filtered_code(worse.code), better_code=filtered_code(better.code),
                        reflection=reflection, func_name="heuristics")
                    crossover_specs.append((f"crossover_{index+1:02d}", (worse, better),
                                            f"{system_generator}\n{user}\n{output_contract}"))
                crossed = self.batch(crossover_specs)
                population = crossed
                self.update_elite(population)
                long_prompt = self.prompts["user_reflector_lt"].format(
                    problem_desc=self.contract.description, prior_reflection=self.long_reflection,
                    new_reflection="\n".join(reflections))
                self.long_reflection = self.request("long_reflection", f"{system_reflector}\n{long_prompt}", plain_response)
                mutation_prompt = self.prompts["mutation"].format(
                    user_generator=user_generator, reflection=self.long_reflection + self.external_knowledge,
                    func_signature1=signature1, elitist_code=filtered_code(self.elite.code), func_name="heuristics")
                mutations = self.batch([(f"mutation_{i+1:02d}", (self.elite,),
                                         f"{system_generator}\n{mutation_prompt}\n{output_contract}")
                                        for i in range(self.mutation_count)])
                population.extend(mutations)
                self.update_elite(population)
                self.event("selection", generation=generation,
                           population=[item.candidate_id for item in population], elite=self.elite.candidate_id,
                           short_reflections=reflections, long_reflection=self.long_reflection)
            self.event("finish", status="completed", elite=self.elite.candidate_id)
            return self.elite
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
            raise StopRun(f"ReEvo 运行意外停止：{type(exc).__name__}: {exc}") from exc
