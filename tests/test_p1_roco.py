from contextlib import ExitStack, nullcontext
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from baselines.roco.paper_controller import RoCoController
from baselines.roco.runtime import RoCoRuntime
from scripts import p1_run_generation, p1_run_roco
from scripts import p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, SOLVER_TIME_SCOPE, timing_description
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog
from tests import test_p1_runtime as runtime_tests


class RoCoBackend(FixtureBackend):
    def __init__(self, contract):
        super().__init__()
        self.contract = contract

    def model(self, stage, prompt):
        response = super().model(stage, prompt)
        return response.replace("def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:",
                                self.contract.signature).replace("np.ones_like(distance_matrix)", "np.array([1.0])")

    def evaluate(self, candidate):
        result = super().evaluate(candidate)
        if self.contract.direction == "max":
            result["fitness"] = result["average"]
        return result


def normalize_role_wording(prompt, proposed, roco):
    """只替换两种角色的称呼、职责与融合措辞，用于检查共同模板。"""
    for key in ("round_priorities", "first_final_priority", "second_final_priority", "balance"):
        prompt = prompt.replace(proposed.fusion_wording[key], roco.fusion_wording[key])
    for key in ("first_final_label", "second_final_label"):
        prompt = prompt.replace(f"Best {proposed.fusion_wording[key]}'s code:",
                                f"Best {roco.fusion_wording[key]}'s code:")
    for proposed_role, roco_role in zip(proposed.design_roles, roco.design_roles):
        own, other = proposed.role_details[proposed_role], roco.role_details[roco_role]
        prompt = prompt.replace(own["identity"], other["identity"])
        prompt = prompt.replace(own["direction"].format(focus=own["focus"]),
                                other["direction"].format(focus=other["focus"]))
        prompt = prompt.replace(own["focus"], other["focus"])
        prompt = prompt.replace(own["label"], other["label"])
    return prompt


class TestRoCo(unittest.TestCase):
    def test_roco_paper_prompts_and_disclosed_shared_clarifications(self):
        paper = (Path(__file__).resolve().parents[1] / "vault/References/RoCo/mineru/full.md").read_text(
            encoding="utf-8").split("## Prompts of RoCo", 1)[1]
        blocks = [match.group(1) for match in re.finditer(r"```txt\s*\n(.*?)\n```", paper, re.S)]

        def fill(template, **values):
            for key, value in values.items():
                template = template.replace("{" + key + "}", str(value))
            return " ".join(template.split())

        contract = load_task_contract("tsp_aco")
        roco = RoCoController(None, None, task_contract=contract)
        q = Candidate("q", 1, "fixture", (), thought="Q_THOUGHT", code="Q_CODE", objective=5, fitness=-5)
        e = Candidate("e", 2, "fixture", (), thought="E_THOUGHT", code="E_CODE", objective=4, fitness=-4)
        common = {"task_description": roco.task, "output_request": roco.output}
        critic = {"advice": "ADVICE", "reflection": "REFLECTION"}
        for role, source in (("explorer", blocks[0] + "\n" + blocks[1]), ("exploiter", blocks[2])):
            expected = fill(source, **common, alg_description=q.thought, code=q.code, obj=q.objective,
                            cri_response=critic["advice"], reflection=critic["reflection"])
            self.assertEqual(expected, " ".join(roco.branch_prompt(role, q, critic).split()))
        expected = fill(blocks[3], **common, explorer_algorithm=q.thought, explorer_code=q.code,
                        explorer_obj=q.objective, exploiter_algorithm=e.thought, exploiter_code=e.code,
                        exploiter_obj=e.objective)
        evidence_limit = "Do not claim measured runtime, quality preservation or causality beyond the supplied evidence.\n"
        actual = roco.round_fusion_prompt([q, e], "Lower")
        self.assertIn(evidence_limit, actual)
        self.assertEqual(expected, " ".join(actual.replace(evidence_limit, "").split()))
        expected = fill(blocks[4], **common, explorer_code=q.code, explorer_obj=q.objective,
                        exploiter_code=e.code, exploiter_obj=e.objective, summary="REFLECTION")
        actual = roco.final_fusion_prompt([q, e], ["REFLECTION"])
        self.assertIn(evidence_limit, actual)
        self.assertEqual(expected, " ".join(actual.replace(evidence_limit, "").split()))
        for value in (4, 6):
            e.objective, e.fitness = value, -value
            actual = roco.branch_critic_prompt(q, e)
            for item in (roco.task, q.thought, q.code, str(q.objective), e.thought, e.code, str(e.objective),
                         "<ref>", "<ans>"):
                self.assertIn(item, actual)
            self.assertIn("do not assume either is the other's parent", actual)
            self.assertNotIn("prior code", actual.lower())
            self.assertNotIn("current code", actual.lower())
        changes = {"explorer": [{"from_objective": 5, "proposal_objective": 4, "to_objective": 4}]}
        actual = roco.long_reflection_prompt("explorer", ["REFLECTION"], changes)
        actual = actual.replace(f"Objective changes: {changes['explorer']}\n", "")
        caution = ("Treat causal explanations as hypotheses; a combined score does not establish "
                   "separate runtime or quality changes. ")
        self.assertIn(caution, actual)
        actual = actual.replace(caution, "")
        expected = fill(blocks[10], task_description=roco.task, reflection="REFLECTION", role="Explorer",
                        role_description=roco.role_focus("explorer"))
        self.assertEqual(expected, " ".join(actual.split()))

        backend, log = RoCoBackend(contract), MemoryLog()
        roco = RoCoController(backend, log, population_size=2, initial_count=6, task_contract=contract)
        roco.run()
        by_id = {candidate.candidate_id: candidate for candidate in roco.candidates}
        pair = next(event for event in log.items if event["event"] == "elite_pair")
        a, b = by_id[pair["a"]], by_id[pair["b"]]
        calls = dict(backend.calls)
        initial = calls["critic_initial"]
        for item in (roco.task, a.code, b.code, str(a.objective), str(b.objective), "<ref>", "<ans>"):
            self.assertIn(item, initial)
        self.assertIn("without assuming a parent-child relationship", initial)
        expected = fill(blocks[11], **common, history_reflection="", reflection="Retain useful changes.",
                        elitist_code=a.code)
        self.assertEqual(expected, " ".join(calls["mutation_A_explorer"].split()))

    def test_shared_templates_with_role_specific_names_and_objectives(self):
        contract = load_task_contract("tsp_aco")
        proposed = Controller(None, None, task_contract=contract)
        roco = RoCoController(None, None, task_contract=contract)
        q = Candidate("q", 1, "fixture", (), thought="first idea", code="first code",
                      objective=5.0, fitness=-5.0, raw_objective=5.0,
                      timing_seconds=2.0, timing_scope=SOLVER_TIME_SCOPE)
        candidates = [Candidate("e", 2, "fixture", (), thought="second idea", code="second code",
                                objective=value, fitness=-value, raw_objective=value,
                                timing_seconds=3.0, timing_scope=SOLVER_TIME_SCOPE) for value in (4.0, 5.0, 6.0)]
        for candidate in candidates:
            self.assertIn("<quality_ref>", proposed.branch_critic_prompt(q, candidate))
            self.assertIn("measured timing", proposed.branch_critic_prompt(q, candidate))
            self.assertIn("<ref>", roco.branch_critic_prompt(q, candidate))
            self.assertNotIn("measured timing", roco.branch_critic_prompt(q, candidate))
        pair = [q, candidates[0]]
        prompt_pairs = [
            (proposed.round_fusion_prompt(pair, "Lower"), roco.round_fusion_prompt(pair, "Lower")),
            (proposed.final_fusion_prompt(pair, ["reflection"]), roco.final_fusion_prompt(pair, ["reflection"])),
            (proposed.merge_reflections_prompt(dict(zip(proposed.design_roles, ("first lesson", "second lesson")))),
             roco.merge_reflections_prompt(dict(zip(roco.design_roles, ("first lesson", "second lesson"))))),
        ]
        critic = {"advice": "feedback", "reflection": "reflection"}
        for proposed_role, roco_role in zip(proposed.design_roles, roco.design_roles):
            own_prompt = proposed.branch_prompt(proposed_role, q, critic)
            if proposed_role == "speed_agent":
                start = own_prompt.index("\nYour current candidate's mean raw objective:")
                end = own_prompt.index("\nReflection:", start)
                evidence = own_prompt[start:end]
                self.assertIn(timing_description(q), evidence)
                self.assertIn("benefit per solve second: 0.1", evidence)
                self.assertIn("A single measurement can be affected by concurrent load", evidence)
                self.assertNotIn("benefit per solve second", roco.branch_prompt(roco_role, q, critic))
                own_prompt = own_prompt[:start] + own_prompt[end:]
            prompt_pairs.extend([
                (own_prompt, roco.branch_prompt(roco_role, q, critic)),
                (proposed.long_reflection_prompt(proposed_role, ["reflection"], {proposed_role: [1]}),
                 roco.long_reflection_prompt(roco_role, ["reflection"], {roco_role: [1]})),
            ])
        for own, baseline in prompt_pairs:
            self.assertEqual(normalize_role_wording(own, proposed, roco), baseline)
            for old in ("Explorer", "Exploiter", "long-term innovation", "short-term refinement"):
                self.assertNotIn(old.lower(), own.lower())

    def test_small_batch_shared_prompts_and_operators_on_identical_responses(self):
        contract = load_task_contract("tsp_aco")
        runs = []
        for controller_class in (Controller, RoCoController):
            backend, log = RoCoBackend(contract), MemoryLog()
            controller = controller_class(backend, log, population_size=2, initial_count=6,
                                          rounds=3, task_contract=contract, seed=2026110601)
            controller.run()
            runs.append((controller, dict(backend.calls), log.items))
        (proposed, first, proposed_events), (roco, second, _) = runs
        self.assertEqual(len(first), 39)
        self.assertEqual(len(second), 39)
        self.assertIn("Solution Quality Agent", first["start_solution_quality_agent"])
        self.assertIn("Speed Agent", first["start_speed_agent"])
        for stage in ("fusion_1", "fusion_2", "fusion_3", "fusion_final", "reflection_integrator"):
            self.assertIn("Solution Quality Agent", first[stage])
            self.assertIn("Speed Agent", first[stage])
        roles = {event.get("source_role") for event in proposed_events if event["event"] == "candidate_registered"}
        self.assertTrue({"solution_quality_agent", "speed_agent", "integrator"} <= roles)
        self.assertEqual({event["role"] for event in proposed_events if event["event"] == "branch_transition"},
                         {"solution_quality_agent", "speed_agent"})
        for stage in sorted(first):
            with self.subTest(stage=stage):
                prompt = first[stage]
                for old in ("Explorer", "Exploiter", "long-term innovation", "short-term refinement",
                            "exploration and exploitation", "creative, future-oriented",
                            "conservative, efficiency-driven"):
                    self.assertNotIn(old.lower(), prompt.lower())
                self.assertNotRegex(stage, r"(?:^|_)(?:Q|E)(?:_|$)")
                counterpart = stage
                for own_role, baseline_role in zip(proposed.design_roles, roco.design_roles):
                    counterpart = counterpart.replace(own_role, baseline_role)
                self.assertIn(counterpart, second)
                changed_feedback = (stage.startswith("critic_") or stage.startswith("start_")
                                    or stage.startswith("round_") or stage == "fusion_final"
                                    or stage in ("reflection_solution_quality_agent", "reflection_speed_agent"))
                if not changed_feedback:
                    self.assertEqual(normalize_role_wording(prompt, proposed, roco), second[counterpart])
        self.assertIn("QUALITY_ADVICE", first["round_1_solution_quality_agent"])
        self.assertNotIn("SPEED_ADVICE", first["round_1_solution_quality_agent"])
        self.assertIn("SPEED_ADVICE", first["round_1_speed_agent"])
        self.assertNotIn("QUALITY_ADVICE", first["round_1_speed_agent"])

    def test_initial_critic_does_not_claim_a_tied_pair_is_better(self):
        contract = load_task_contract("tsp_aco")

        class TiedBackend(RoCoBackend):
            def evaluate(self, candidate):
                super().evaluate(candidate)
                return {"status": "passed", "average": 1.0, "fitness": -1.0,
                        "average_solve_wall_seconds": 1.0, "start": {"time_scope": SOLVER_TIME_SCOPE}}

        for controller_class in (Controller, RoCoController):
            with self.subTest(controller=controller_class.__name__):
                backend, log = TiedBackend(contract), MemoryLog()
                controller_class(backend, log, population_size=2, initial_count=6,
                                 rounds=3, task_contract=contract).run()
                prompt = dict(backend.calls)["critic_initial"]
                self.assertIn("The objective scores are tied", prompt)
                self.assertNotIn("Better code:", prompt)
                self.assertNotIn("Worse code:", prompt)

    def test_small_batch_runs_all_required_stages_with_two_parents(self):
        contract = load_task_contract("tsp_aco")
        for controller_class in (Controller, RoCoController):
            with self.subTest(controller=controller_class.__name__):
                backend, log = RoCoBackend(contract), MemoryLog()
                controller = controller_class(backend, log, population_size=2, initial_count=6,
                                              rounds=3, task_contract=contract)
                selected = controller.run()
                self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (39, 32, 2))
                for operator in ("E1", "E2", "M1", "M2"):
                    proposals = [candidate for candidate in controller.candidates
                                 if candidate.stage.startswith(f"eoh_{operator}_")]
                    self.assertEqual(len(proposals), 2)
                    self.assertTrue(all(len(c.parents) == (2 if operator in ("E1", "E2") else 1)
                                        for c in proposals))
                stages = [stage for stage, _ in backend.calls]
                self.assertEqual(sum(stage.startswith("init_") for stage in stages), 6)
                self.assertEqual(sum(stage.startswith("mutation_") for stage in stages), 6)
                for turn in (1, 2, 3):
                    for stage in (f"critic_{turn}", *(f"round_{turn}_{role}" for role in controller.design_roles),
                                  f"fusion_{turn}"):
                        self.assertIn(stage, stages)

    def test_all_conditions_use_original_role_goals_and_record_mapping(self):
        conditions = [(name, black_box) for name in sorted(TASKS)
                      for black_box in ((False,) if name == "tsp_gls" else (False, True))]
        for name, black_box in conditions:
            with self.subTest(name=name, black_box=black_box):
                contract = load_task_contract(name, black_box=black_box)
                backend, log = RoCoBackend(contract), MemoryLog()
                selected = RoCoController(backend, log, task_contract=contract).run()
                self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (95, 88, 10))
                calls = dict(backend.calls)
                self.assertIn("global diversity and long-term potential", calls["start_explorer"])
                self.assertIn("short-term gains and local optimizations", calls["start_exploiter"])
                self.assertNotIn("reducing computational cost", calls["start_exploiter"])
                self.assertIn("As an optimization expert", calls["critic_1"])
                self.assertIn("Exploiter", calls["fusion_1"])
                self.assertIn("Best explorer's code", calls["fusion_final"])
                self.assertIn("for Exploiter short-term gains", calls["reflection_exploiter"])
                self.assertIn("Explorer:", calls["reflection_integrator"])
                self.assertEqual(log.items[0]["internal"], {"explorer": "Explorer", "exploiter": "Exploiter", "integrator": "Integrator"})
                roles = {event.get("source_role") for event in log.items if event["event"] == "candidate_registered"}
                self.assertTrue({"Explorer", "Exploiter", "Integrator"} <= roles)
                self.assertNotIn("explorer", roles)
                self.assertNotIn("exploiter", roles)

    def test_two_generations_retain_role_specific_histories(self):
        contract = load_task_contract("tsp_aco")
        backend, log = RoCoBackend(contract), MemoryLog()
        controller = RoCoController(backend, log, task_contract=contract, generations=2)
        controller.run()
        self.assertEqual((len(backend.calls), len(backend.evaluations)), (160, 146))
        self.assertEqual([len(controller.history[key]) for key in ("explorer", "exploiter", "integrator")], [2, 2, 2])
        self.assertEqual([event["role"] for event in log.items if event["event"] == "history_appended"],
                         ["Explorer", "Exploiter", "Integrator"] * 2)


class TestRoCoRunGate(unittest.TestCase):
    def test_entrypoints_pass_frozen_population_to_controllers(self):
        frozen, _ = matrix.load_matrix(allow_pending=True)
        for module, method, runtime_name, controller_name in (
                (p1_run_generation, "proposed_solution_quality_and_speed_agents", "TaskRuntime", "Controller"),
                (p1_run_roco, "roco_paper_adapted", "RoCoRuntime", "RoCoController")):
            with self.subTest(method=method), ExitStack() as stack:
                config = matrix.training_config(frozen, "tsp_aco", "white_box", method, 1)
                runtime = SimpleNamespace(protected_hashes={}, contract=load_task_contract("tsp_aco"), cli=None)
                controller = Mock()
                stack.enter_context(patch.object(Path, "read_text", return_value=json.dumps(config)))
                stack.enter_context(patch.object(Path, "mkdir"))
                stack.enter_context(patch.object(Path, "open", return_value=io.StringIO()))
                stack.enter_context(patch.object(sys, "argv", ["runner", "--config", "fixture", "--run"]))
                stack.enter_context(patch.object(module, "verify_training_config"))
                stack.enter_context(patch.object(module, "JsonLog"))
                stack.enter_context(patch.object(module, runtime_name, return_value=runtime))
                stack.enter_context(patch.object(module, "memory_and_priority", return_value={
                    "available_memory_bytes": config["resources"]["min_available_system_memory_bytes"]}))
                stack.enter_context(patch.object(module.probe, "cli_command", return_value=["fixture-cli"]))
                probe = stack.enter_context(patch.object(module.probe, "run_command", side_effect=[
                    {"exit_code": 0, "stdout": "fixture"},
                    {"exit_code": 0, "stdout": "Logged in using ChatGPT"}]))
                stack.enter_context(patch("scripts.p0_budget_quota.read_limits",
                                          side_effect=AssertionError("unexpected quota probe")))
                stack.enter_context(patch.object(module, "formal_run_guard", return_value=nullcontext()))
                constructor = stack.enter_context(patch.object(module, controller_name, return_value=controller))
                stack.enter_context(patch("builtins.print"))
                self.assertEqual(module.main(), 0)
                self.assertEqual(constructor.call_args.kwargs["population_size"], 10)
                self.assertEqual(constructor.call_args.kwargs["initial_count"], 30)
                self.assertEqual(constructor.call_args.kwargs["rounds"], 3)
                controller.run.assert_called_once_with()
                self.assertEqual(probe.call_count, 2)  # 模拟版本与登录检查；没有模型请求。

    def test_roco_and_proposed_method_configs_cannot_be_interchanged(self):
        config = runtime_tests.TestFormalRunGate.config()
        config["method"] = {"id": "roco_paper_adapted"}
        self.assertEqual(p1_run_roco.validate_config(config), (6, 20261103))
        with self.assertRaises(ValueError):
            p1_run_generation.validate_config(config)
        runtime = runtime_tests.TestTaskRuntime.create(config, runtime_class=RoCoRuntime)
        self.assertTrue(any("baselines\\roco\\paper_controller.py" in str(path) for path in runtime.protected))
        config["method"]["id"] = "proposed_solution_quality_and_speed_agents"
        with self.assertRaises(ValueError):
            p1_run_roco.validate_config(config)


if __name__ == "__main__":
    unittest.main()
