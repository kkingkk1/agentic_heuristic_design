"""固定来源的 HSEvo 适配：反思、交叉、变异和有记录的和声搜索。"""

import ast
import math
from pathlib import Path
import re
import time

import numpy as np

from baselines.reevo.controller import ReEvoController, filtered_code, parse_generated, plain_response, source_files as reevo_files, text
from scripts import p0_model_probe as probe
from scripts.p0_controller import Candidate, StopRun, check_modification_scope


COMMON = Path(__file__).resolve().parent / "upstream" / "prompts" / "common"
SCIENTISTS = (
    "You are an expert in the domain of optimization heuristics.",
    "You are Albert Einstein, relativity theory developer.",
    "You are Isaac Newton, the father of physics.",
    "You are Marie Curie, pioneer in radioactivity.",
    "You are Nikola Tesla, master of electricity.",
    "You are Galileo Galilei, champion of heliocentrism.",
    "You are Stephen Hawking, black hole theorist.",
    "You are Richard Feynman, quantum mechanics genius.",
    "You are Rosalind Franklin, DNA structure revealer.",
    "You are Ada Lovelace, computer programming pioneer.",
)


def source_files(contract):
    names = ("system_generator", "user_generator", "seed", "user_flash_reflection",
             "user_comprehensive_reflection", "system_reflector", "crossover", "mutation",
             "system_harmony_search", "harmony_search")
    files = {name: COMMON / f"{name}.txt" for name in names}
    shared = reevo_files(contract)
    files["seed_func"] = shared["seed_func"]
    if "external_knowledge" in shared:
        files["external_knowledge"] = shared["external_knowledge"]
    return files


def flash_response(response):
    response = plain_response(response)
    match = re.search(r"\*\*Analysis:\*\*(.*?)\*\*Experience:\*\*(.*)", response, flags=re.DOTALL)
    if not match or not match.group(1).strip() or not match.group(2).strip():
        raise probe.ProbeError("HSEvo 快速反思缺少 Analysis 或 Experience。")
    return {"analyze": match.group(1).strip(), "exp": match.group(2).strip()}


def harmony_response(contract, response):
    """只解析数字字面量；不执行模型生成的参数字典。"""
    if not isinstance(response, str):
        raise probe.ProbeError("HSEvo 和声搜索输出不是字符串。")
    blocks = re.findall(r"```python\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if len(blocks) != 2:
        raise probe.ProbeError("HSEvo 和声搜索须返回函数和参数范围两个代码块。")
    try:
        tree = ast.parse(blocks[0].strip())
        ranges_module = ast.parse(blocks[1].strip())
        if len(ranges_module.body) != 1 or not isinstance(ranges_module.body[0], ast.Assign):
            raise ValueError("参数范围须为单一赋值。")
        assignment = ranges_module.body[0]
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "parameter_ranges":
            raise ValueError("参数范围变量名无效。")
        ranges = ast.literal_eval(assignment.value)
        if not isinstance(ranges, dict) or not ranges:
            raise ValueError("参数范围为空。")
        for name, interval in ranges.items():
            if (not isinstance(name, str) or not name.isidentifier() or not isinstance(interval, (tuple, list))
                    or len(interval) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                                or not math.isfinite(v) for v in interval)
                    or interval[0] >= interval[1]):
                raise ValueError("参数名称或有限范围无效。")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "heuristics_v2"]
        if len(functions) != 1:
            raise ValueError("和声搜索函数身份无效。")
        function = functions[0]
        args = function.args
        actual = tuple((arg.arg, ast.unparse(arg.annotation) if arg.annotation else None)
                       for arg in args.args[:len(contract.parameters)])
        extras = args.args[len(contract.parameters):]
        if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or actual != contract.parameters
                or [arg.arg for arg in extras] != list(ranges) or len(args.defaults) != len(extras)
                or function.returns is None or ast.unparse(function.returns) != "np.ndarray"):
            raise ValueError("参数化函数与任务接口不符。")
        for default in args.defaults:
            value = ast.literal_eval(default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("参数默认值须为有限数字。")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise probe.ProbeError(f"HSEvo 和声搜索输出无效：{exc}") from exc
    return blocks[0].strip(), ranges


def instantiate(contract, template, ranges, vector):
    """把新增默认参数转为函数内常量，保持冻结接口与按位置调用语义。"""
    if len(vector) != len(ranges):
        raise ValueError("和声向量维度不符。")
    tree = ast.parse(template)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "heuristics_v2")
    function.args.args = function.args.args[:len(contract.parameters)]
    function.args.defaults = []
    assignments = [ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(float(value)))
                   for name, value in zip(ranges, vector)]
    index = 1 if (function.body and isinstance(function.body[0], ast.Expr)
                  and isinstance(function.body[0].value, ast.Constant)
                  and isinstance(function.body[0].value.value, str)) else 0
    function.body[index:index] = assignments
    code = ast.unparse(ast.fix_missing_locations(tree))
    return parse_generated(contract, code)["code"]


class HSEvoController(ReEvoController):
    """保留固定来源的核心机制，并标明接口、种子和错误处理适配。"""

    def __init__(self, backend, log, contract, *, population_size=10, initial_count=30, mutation_count=5,
                 generations=1, seed=20261103, hm_size=5, hs_iterations=5, hmcr=0.7, par=0.5, bandwidth=0.2):
        super().__init__(backend, log, contract, population_size=population_size, initial_count=initial_count,
                         mutation_count=mutation_count, generations=generations, seed=seed)
        if (type(hm_size) is not int or hm_size < 1 or type(hs_iterations) is not int or hs_iterations < 0
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in (hmcr, par, bandwidth))):
            raise ValueError("HSEvo 和声搜索参数无效。")
        self.hm_size, self.hs_iterations = hm_size, hs_iterations
        self.hmcr, self.par, self.bandwidth = hmcr, par, bandwidth
        self.files = source_files(contract)
        self.prompts = {name: text(path) for name, path in self.files.items()}
        self.external_knowledge = self.prompts.get("external_knowledge", "")
        self.good, self.bad = [], []

    def register(self, stage, parents):
        with self._lock:
            sequence = len(self.candidates) + 1
            candidate = Candidate(f"c{sequence:04d}", sequence, stage, parents)
            self.candidates.append(candidate)
        self.event("candidate_registered", candidate_id=candidate.candidate_id, sequence=sequence,
                   stage=stage, source_role="HSEvo", operator=stage.split("_", 1)[0],
                   generation=self.generation, round=None, parents=list(parents))
        return candidate

    def user_generator(self, scientist):
        return self.prompts["user_generator"].format(seed=scientist, func_name="heuristics",
                                                     problem_desc=self.contract.description,
                                                     func_desc=self.contract.function_description)

    def system_generator(self, scientist):
        return self.prompts["system_generator"].format(seed=scientist)

    def initial_prompt(self, index):
        scientist = SCIENTISTS[index % len(SCIENTISTS)]
        seed_prompt = self.prompts["seed"].format(seed_func=self.prompts["seed_func"], func_name="heuristics")
        output_contract = f"Use the exact interface {self.contract.signature} and include import numpy as np."
        return f"{self.system_generator(scientist)}\n{self.user_generator(scientist)}\n{seed_prompt}\n{output_contract}"

    def evaluate_harmony(self, template, ranges, vector, parent, index):
        candidate = self.register(f"harmony_{index:02d}", (parent.candidate_id,))
        candidate.code = instantiate(self.contract, template, ranges, vector)
        candidate.thought = "HSEvo harmony-search parameters"
        check_modification_scope(candidate.code)
        self._generated += 1
        candidate.generation_order = self._generated
        self.event("harmony_candidate", candidate_id=candidate.candidate_id, parent=parent.candidate_id,
                   parameters=dict(zip(ranges, map(float, vector))), code=candidate.code)
        self.backend.prepare_proposal(candidate)
        try:
            result = self.backend.evaluate(candidate)
        finally:
            self.backend.release_proposal(candidate)
        self.event("evaluation", candidate_id=candidate.candidate_id, result=result)
        if result.get("status") == "infrastructure_error":
            raise StopRun(f"和声搜索评价基础设施不可信：{result.get('error')}")
        if result.get("status") != "passed":
            candidate.error = str(result.get("error") or result.get("status"))
            return candidate
        objective, fitness = result.get("average"), result.get("fitness")
        if (isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective)
                or isinstance(fitness, bool) or not isinstance(fitness, (int, float)) or not math.isfinite(fitness)
                or fitness != (-objective if self.direction == "min" else objective)):
            raise StopRun("和声搜索评价目标或 Fitness 不可信。")
        candidate.objective, candidate.fitness = float(objective), float(fitness)
        candidate.completed_order, candidate.evaluation_origin = time.monotonic_ns(), candidate.candidate_id
        return candidate

    def harmony_search(self, population):
        valid = [candidate for candidate in population if candidate.valid]
        if not valid:
            raise StopRun("HSEvo 没有可供和声搜索的有效个体。")
        parent = max(valid, key=lambda candidate: candidate.fitness)
        prompt = self.prompts["system_harmony_search"] + "\n" + self.prompts["harmony_search"].format(code_extract=parent.code)
        template, ranges = self.request("harmony_parameters", prompt, lambda r: harmony_response(self.contract, r))
        memory = np.zeros((self.hm_size, len(ranges)))
        for i, (lower, upper) in enumerate(ranges.values()):
            memory[:, i] = self.random_state.uniform(lower, upper, self.hm_size)
        evaluated = [self.evaluate_harmony(template, ranges, vector, parent, i + 1)
                     for i, vector in enumerate(memory)]
        if not any(item.valid for item in evaluated):
            raise StopRun("HSEvo 和声初始化均无有效评价。")
        for iteration in range(self.hs_iterations):
            vector = np.zeros(len(ranges))
            for i, (lower, upper) in enumerate(ranges.values()):
                if self.random_state.rand() < self.hmcr:
                    vector[i] = memory[self.random_state.randint(0, self.hm_size), i]
                    if self.random_state.rand() < self.par:
                        vector[i] += self.random_state.uniform(-1, 1) * (upper - lower) * self.bandwidth
                else:
                    vector[i] = self.random_state.uniform(lower, upper)
            new = self.evaluate_harmony(template, ranges, vector, parent, self.hm_size + iteration + 1)
            worst = min(range(len(evaluated)), key=lambda i: evaluated[i].fitness if evaluated[i].valid else -math.inf)
            if new.valid and (not evaluated[worst].valid or new.fitness > evaluated[worst].fitness):
                evaluated[worst], memory[worst] = new, vector
        best = max((item for item in evaluated if item.valid), key=lambda item: item.fitness)
        self.event("harmony_selection", parent=parent.candidate_id, selected=best.candidate_id)
        return best

    def run(self):
        try:
            seed = self.evaluate_seed()
            seed.thought = "HSEvo task-adapted reference seed"
            output_contract = f"Use the exact interface {self.contract.signature} and include import numpy as np."
            initial_specs = []
            for i in range(self.initial_count):
                initial_specs.append((f"init_{i+1:02d}", (), self.initial_prompt(i)))
            population = self.batch(initial_specs)
            self.update_elite(population)
            self.event("initial_population", candidate_ids=[item.candidate_id for item in population],
                       seed_candidate_id=self.seed.candidate_id)
            for generation in range(1, self.generations + 1):
                self.generation = generation
                self.backend.before_generation()
                pairs = self.parent_pairs(population)
                selected = [item for pair in pairs for item in pair]
                ranked = sorted(selected, key=lambda item: -item.fitness)
                methods = "\n".join(f"[Heuristics {i+1}]\n{item.code}\n" for i, item in enumerate(ranked))
                flash = self.prompts["user_flash_reflection"].format(problem_desc=self.contract.description,
                          lst_method=methods, schema_reflection={"analyze": "str", "exp": "str"})
                insight = self.request("flash_reflection", f"{self.prompts['system_reflector']}\n{flash}", flash_response)
                comprehensive = self.prompts["user_comprehensive_reflection"].format(
                    good_reflection="\n\n".join(self.good) if self.good else "None",
                    bad_reflection="\n\n".join(self.bad) if self.bad else "None", curr_reflection=insight["exp"])
                memory = self.request("comprehensive_reflection", f"{self.prompts['system_reflector']}\n{comprehensive}", plain_response)
                memory = self.external_knowledge + "\n" + memory
                former_elite = self.elite.candidate_id
                signature0 = self.contract.signature.replace("heuristics_v2", "heuristics_v0")
                signature1 = self.contract.signature.replace("heuristics_v2", "heuristics_v1")
                user = self.user_generator(SCIENTISTS[0])
                crossover_specs = []
                for i, (worse, better) in enumerate(pairs):
                    prompt = self.prompts["crossover"].format(user_generator=user, func_signature_m1=signature0,
                        func_signature_m2=signature1, code_method1=filtered_code(better.code),
                        code_method2=filtered_code(worse.code), analyze=insight["analyze"], exp=memory,
                        func_name="heuristics")
                    crossover_specs.append((f"crossover_{i+1:02d}", (better, worse),
                                            f"{self.system_generator(SCIENTISTS[0])}\n{prompt}\n{output_contract}"))
                population = self.batch(crossover_specs)
                self.update_elite(population)
                mutation = self.prompts["mutation"].format(user_generator=user, reflection=memory,
                    func_signature1=signature1, elitist_code=filtered_code(self.elite.code), func_name="heuristics")
                population.extend(self.batch([(f"mutation_{i+1:02d}", (self.elite,),
                    f"{self.system_generator(SCIENTISTS[0])}\n{mutation}\n{output_contract}")
                    for i in range(self.mutation_count)]))
                self.update_elite(population)
                (self.good if self.elite.candidate_id != former_elite else self.bad).append(insight["exp"])
                population.append(self.harmony_search(population))
                self.update_elite(population)
                self.event("selection", generation=generation, population=[item.candidate_id for item in population],
                           elite=self.elite.candidate_id, good_reflections=self.good, bad_reflections=self.bad)
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
            raise StopRun(f"HSEvo 运行意外停止：{type(exc).__name__}: {exc}") from exc
