import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from scripts import p0_controller as control
from scripts.p0_runtime import Budget, Runtime, generation_requirements
from scripts.p0_run_generation import quota_gate
from scripts.p0_windows_job import WindowsJob


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "cfg" / "p0_tsp_aco.json").read_text(encoding="utf-8"))


def small_config():
    config = copy.deepcopy(CONFIG)
    config["task"].update(population_size=2, initial_proposals=6)
    config["budget"].update(normal_model_calls=39, normal_code_generation_slots=32,
                            additional_infrastructure_retry_calls=0,
                            additional_infrastructure_evaluation_starts=0,
                            max_initialization_and_generation_model_cli_invocations=39,
                            max_initialization_and_generation_evaluation_starts=32,
                            future_initialization_and_generation_eval_process_tree_cpu_seconds=18000)
    return config


class MemoryLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = []

    def event(self, kind, **fields):
        with self.lock:
            self.items.append({"event": kind, **fields})


class FixtureBackend:
    def __init__(self, *, invalid=(), runtime_invalid=(), duplicate_initial=False, duplicate_generation=False):
        self.invalid = set(invalid)
        self.runtime_invalid = set(runtime_invalid)
        self.duplicate_initial = duplicate_initial
        self.duplicate_generation = duplicate_generation
        self.calls = []
        self.evaluations = []
        self.generation_checks = 0
        self.lock = threading.Lock()

    def model(self, stage, prompt):
        with self.lock:
            self.calls.append((stage, prompt))
        if stage.startswith("critic"):
            return "<ref>Keep the useful features.</ref><ans>Improve the edge scores.</ans>"
        if stage == "reflection_integrator":
            return "Combine both lessons."
        if stage.startswith("reflection_"):
            return "<ans>Retain useful changes.</ans>"
        if stage in self.invalid:
            return "<thought>bad</thought><code>invalid syntax !!!</code>"
        marker = (stage.replace("solution_quality_agent", "first_role").replace("explorer", "first_role")
                  .replace("speed_agent", "second_role").replace("exploiter", "second_role"))
        if self.duplicate_initial and stage.startswith("init_"):
            marker = "identical_initial"
        if self.duplicate_generation and not stage.startswith("init_"):
            marker = "identical_generation"
        return (f"<thought>Idea {marker}.</thought><code>import numpy as np\n"
                f"def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:\n"
                f"    return np.ones_like(distance_matrix)  # {marker}\n</code>")

    def prepare_proposal(self, candidate):
        pass

    def release_proposal(self, candidate):
        pass

    def evaluate(self, candidate):
        with self.lock:
            self.evaluations.append(candidate.stage)
        if candidate.stage in self.runtime_invalid:
            return {"status": "candidate_error", "error": "fixture runtime error"}
        # 第 2 轮第一分支分数下降，但仍是有效提案，必须续写。
        objective = (2000.0 if candidate.stage in ("round_2_solution_quality_agent", "round_2_explorer")
                     else float(1000 - candidate.sequence))
        return {"status": "passed", "average": objective, "fitness": -objective}

    def before_generation(self):
        self.generation_checks += 1


class ControllerTests(unittest.TestCase):
    def make(self, **kwargs):
        self.backend, self.log = FixtureBackend(**kwargs), MemoryLog()
        return control.Controller(self.backend, self.log)

    def test_normal_generation_counts_slots_fixed_pool_and_prompt_boundaries(self):
        controller = self.make()
        selected = controller.run()
        self.assertEqual(len(self.backend.calls), 95)
        self.assertEqual(len(self.backend.evaluations), 88)
        self.assertEqual(len(selected), 10)
        self.assertEqual(self.backend.generation_checks, 1)
        self.assertEqual([len(controller.history[r]) for r in ("solution_quality_agent", "speed_agent", "integrator")], [1, 1, 1])
        calls = dict(self.backend.calls)
        self.assertNotIn("return np.", calls["init_01"])
        self.assertIn("heuristics_v2(distance_matrix: np.ndarray)", calls["init_01"])
        self.assertNotIn("Objective score:", calls["eoh_E1_01"])
        pair = next(item for item in self.log.items if item["event"] == "elite_pair")
        registered = next(item for item in self.log.items if item["event"] == "candidate_registered"
                          and item["stage"] == "round_1_solution_quality_agent")
        self.assertEqual((registered["source_role"], registered["operator"],
                          registered["generation"], registered["round"]), ("solution_quality_agent", "continuation", 1, 1))
        candidates = {c.candidate_id: c for c in controller.candidates}
        a, b = candidates[pair["a"]], candidates[pair["b"]]
        for role in ("solution_quality_agent", "speed_agent"):
            prompt = calls[f"start_{role}"]
            self.assertIn(b.code, prompt)
            self.assertNotIn(a.code, prompt)
        q1 = next(c for c in controller.candidates if c.stage == "round_1_solution_quality_agent")
        e1 = next(c for c in controller.candidates if c.stage == "round_1_speed_agent")
        self.assertIn(q1.code, calls["round_2_solution_quality_agent"])
        self.assertNotIn(e1.code, calls["round_2_solution_quality_agent"])
        self.assertNotIn("Critic's feedback", calls["fusion_1"])
        self.assertIn("solution quality contribution", calls["fusion_1"])
        self.assertIn("computational efficiency improvement", calls["fusion_final"])
        self.assertIn("Based on the critic's feedback and the reflection", calls["start_solution_quality_agent"])
        self.assertNotIn(q1.thought, calls["fusion_final"])
        self.assertNotIn(a.thought, calls["mutation_A_solution_quality_agent"])
        self.assertNotIn(str(a.objective), calls["mutation_A_solution_quality_agent"])
        transitions = {(x["role"], x["round"]): x for x in self.log.items if x["event"] == "branch_transition"}
        for turn in (1, 2, 3):
            for role in ("solution_quality_agent", "speed_agent"):
                previous = candidates[transitions[role, turn]["previous"]]
                other = candidates[transitions["speed_agent" if role == "solution_quality_agent" else "solution_quality_agent", turn]["previous"]]
                self.assertIn(previous.code, calls[f"round_{turn}_{role}"])
                if previous.code != other.code:
                    self.assertNotIn(other.code, calls[f"round_{turn}_{role}"])
            q = candidates[transitions["solution_quality_agent", turn]["proposal"]]
            e = candidates[transitions["speed_agent", turn]["proposal"]]
            fusion_prompt = calls[f"fusion_{turn}"]
            self.assertIn(q.thought, fusion_prompt)
            self.assertIn(e.thought, fusion_prompt)
            self.assertNotIn("Critic's feedback", fusion_prompt)
        for role in ("solution_quality_agent", "speed_agent"):
            reflection_prompt = calls[f"reflection_{role}"]
            self.assertNotIn("Algorithm description:", reflection_prompt)
            self.assertNotIn("Code:\n", reflection_prompt)
        for elite_name, elite, other in (("A", a, b), ("B", b, a)):
            for role in ("solution_quality_agent", "speed_agent", "integrator"):
                mutation_prompt = calls[f"mutation_{elite_name}_{role}"]
                self.assertIn(elite.code, mutation_prompt)
                self.assertNotIn(elite.thought, mutation_prompt)
                self.assertNotIn(other.code, mutation_prompt)
        selection = next(item for item in self.log.items if item["event"] == "selection")
        self.assertNotIn(a.candidate_id, selection["pool"])
        self.assertNotIn(b.candidate_id, selection["pool"])
        self.assertNotIn(q1.candidate_id, selection["pool"])
        selected_at = self.log.items.index(selection)
        self.assertTrue(all(self.log.items.index(item) > selected_at for item in self.log.items
                            if item["event"] == "history_appended"))

    def test_invalid_proposals_fall_back_skip_fusions_and_do_not_retry(self):
        controller = self.make(invalid={"start_solution_quality_agent", "round_1_solution_quality_agent"}, runtime_invalid={"round_3_speed_agent"})
        controller.run()
        transitions = {(x["role"], x["round"]): x for x in self.log.items if x["event"] == "branch_transition"}
        self.assertEqual(transitions["solution_quality_agent", 0]["current"], transitions["solution_quality_agent", 0]["previous"])
        self.assertEqual(transitions["solution_quality_agent", 1]["current"], transitions["solution_quality_agent", 1]["previous"])
        self.assertEqual(transitions["solution_quality_agent", 2]["current"], transitions["solution_quality_agent", 2]["proposal"])
        self.assertGreater(transitions["solution_quality_agent", 2]["to_objective"], transitions["solution_quality_agent", 2]["from_objective"])
        self.assertEqual(transitions["solution_quality_agent", 3]["previous"], transitions["solution_quality_agent", 2]["proposal"])
        self.assertEqual(transitions["speed_agent", 3]["current"], transitions["speed_agent", 3]["previous"])
        skipped = {x["stage"] for x in self.log.items if x["event"] == "slot_skipped"}
        self.assertEqual(skipped, {"fusion_1", "fusion_3"})
        self.assertNotIn("round_1_solution_quality_agent", self.backend.evaluations)
        self.assertEqual(sum(stage == "round_3_speed_agent" for stage, _ in self.backend.calls), 1)
        selection = next(x for x in self.log.items if x["event"] == "selection")
        self.assertNotIn(transitions["speed_agent", 3]["current"], selection["pool"])

    def test_too_few_initial_or_final_candidates_stops_without_history(self):
        for options in ({"duplicate_initial": True}, {"duplicate_generation": True}):
            with self.subTest(options=options):
                controller = self.make(**options)
                with self.assertRaises(control.StopRun):
                    controller.run()
                self.assertFalse(any(controller.history.values()))
                self.assertFalse(any(x["event"] == "history_appended" for x in self.log.items))
                self.assertEqual(self.log.items[-1]["status"], "stopped")
                if options.get("duplicate_generation"):
                    stages = [stage for stage, _ in self.backend.calls]
                    self.assertIn("fusion_1", stages)
                    self.assertIn("fusion_2", stages)
                    self.assertIn("fusion_3", stages)

    def test_dedup_uses_earliest_complete_evaluation_without_admitting_old_parent(self):
        controller = self.make()
        early = control.Candidate("old", 1, "init_01", (), code="same", objective=10.0,
                                  fitness=-10.0, generation_order=1, completed_order=1)
        new = control.Candidate("new", 2, "eoh_E1_01", (), code="same", objective=1.0,
                                fitness=-1.0, generation_order=2, completed_order=2)
        controller.candidates = [early, new]
        chosen = controller.select([new], 1)[0]
        self.assertEqual(chosen.candidate_id, "new")
        self.assertEqual(chosen.evaluation_origin, "old")
        self.assertEqual(chosen.fitness, -10.0)

    def test_equal_fitness_and_duplicate_code_follow_actual_generation_order(self):
        controller = self.make()
        later = control.Candidate("later", 1, "init_01", (), code="first-code", objective=5.0,
                                  fitness=-5.0, generation_order=2, completed_order=1)
        earlier = control.Candidate("earlier", 2, "init_02", (), code="second-code", objective=5.0,
                                    fitness=-5.0, generation_order=1, completed_order=2)
        controller.candidates = [later, earlier]
        self.assertEqual([c.candidate_id for c in controller.select([later, earlier], 2)], ["earlier", "later"])
        earlier.code = later.code
        selected = controller.select([later, earlier], 1)[0]
        self.assertEqual(selected.candidate_id, "earlier")
        self.assertEqual(selected.evaluation_origin, "later")

    def test_scope_check_rejects_obvious_writes(self):
        for code in ("import os", "open('x','w')", "import numpy as np\nnp.save('x', 1)",
                     "from numpy import save\nsave('x', 1)", "x = getattr(a, 'save')",
                     "import numpy as np\nx = np.save"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                control.check_modification_scope(code)

    def test_critic_instructions_follow_objective_order_and_ties(self):
        q = control.Candidate("q", 1, "round_1_solution_quality_agent", (), thought="q", code="q", objective=5, fitness=-5)
        better = control.Candidate("e", 2, "round_1_speed_agent", (), thought="e", code="e", objective=4, fitness=-4)
        worse = control.Candidate("e", 2, "round_1_speed_agent", (), thought="e", code="e", objective=6, fitness=-6)
        tied = control.Candidate("e", 2, "round_1_speed_agent", (), thought="e", code="e", objective=5, fitness=-5)
        self.assertIn("Current code is better", control.round_critic_prompt(q, better))
        self.assertIn("Better and current code: e", control.round_critic_prompt(q, better))
        self.assertIn("current code is worse", control.round_critic_prompt(q, worse))
        self.assertIn("Worse and current code: e", control.round_critic_prompt(q, worse))
        self.assertIn("objectives are tied", control.round_critic_prompt(q, tied))
        q.fitness = 5
        better.objective = better.fitness = 6
        prompt = control.round_critic_prompt(q, better, direction="max")
        self.assertIn("A higher objective is better", prompt)
        self.assertIn("Better and current code: e", prompt)


class BudgetTests(unittest.TestCase):
    def test_generation_slots_follow_population_size(self):
        self.assertEqual(generation_requirements(2), (33, 26))
        self.assertEqual(generation_requirements(10), (65, 58))
        for invalid in (True, 1, 2.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generation_requirements(invalid)

    def test_small_generation_requires_exact_remaining_slots(self):
        budget = Budget(small_config(), MemoryLog())
        budget.model_normal = budget.eval_starts = budget.eval_normal = 6
        budget.eval_cpu_samples = [1.0]
        budget.before_generation()
        budget.model_normal = 7
        with self.assertRaisesRegex(control.StopRun, "模型调用不足"):
            budget.before_generation()
        budget.model_normal = 6
        budget.eval_starts = budget.eval_normal = 7
        with self.assertRaisesRegex(control.StopRun, "评价启动次数不足"):
            budget.before_generation()

    def test_reserves_across_concurrent_starts_and_counts_completed_children(self):
        log = MemoryLog()
        budget = Budget(CONFIG, log)
        budget.cores = 1
        class Job:
            def __init__(self, cpu):
                self.cpu = cpu
            def accounting(self):
                return {"cpu_seconds": self.cpu}
        jobs = [Job(i + 1) for i in range(4)]
        for i in range(4):
            budget.reserve_proposal(str(i), threading.Event())
        threads = [threading.Thread(target=budget.begin_evaluation, args=(str(i), job))
                   for i, job in enumerate(jobs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(budget.eval_starts, 4)
        self.assertEqual(budget.used_cpu(), 10)
        for i in range(4):
            budget.finish_evaluation(str(i))
            budget.release_proposal(str(i))
        self.assertEqual(budget.used_cpu(), 10)
        budget.cpu_finished = budget.cpu_limit - budget.cores * (budget.timeout + 2) + 1
        with self.assertRaises(control.StopRun):
            budget.reserve_proposal("overflow", threading.Event())

    def test_concurrent_proposals_wait_until_full_evaluation_reservation_is_free(self):
        budget = Budget(CONFIG, MemoryLog())
        budget.cores = 28
        cancelled = threading.Event()
        for identifier in ("a", "b", "c"):
            budget.reserve_proposal(identifier, cancelled)
        acquired = threading.Event()
        def reserve_fourth():
            budget.reserve_proposal("d", cancelled)
            acquired.set()
        thread = threading.Thread(target=reserve_fourth)
        thread.start()
        self.assertFalse(acquired.wait(0.2))
        budget.release_proposal("a")
        self.assertTrue(acquired.wait(2))
        thread.join(timeout=2)
        for identifier in ("b", "c", "d"):
            budget.release_proposal(identifier)

    def test_model_reservation_has_atomic_retry_limit(self):
        budget = Budget(CONFIG, MemoryLog())
        for _ in range(20):
            budget.model_start(retry=True)
        with self.assertRaises(control.StopRun):
            budget.model_start(retry=True)
        self.assertEqual(budget.model_retries, 20)

    def test_live_cpu_guard_terminates_all_active_jobs(self):
        budget = Budget(CONFIG, MemoryLog())
        class Job:
            def __init__(self):
                self.terminated = False
            def accounting(self):
                return {"cpu_seconds": 5990.0}
            def terminate(self):
                self.terminated = True
        job = Job()
        budget.jobs["active"] = job
        with self.assertRaises(control.StopRun):
            budget.check_cpu()
        self.assertTrue(job.terminated)

    def test_quota_cannot_be_inferred_from_window_percentages(self):
        with self.assertRaises(control.StopRun):
            quota_gate({"ordinary_usage_allowed": True, "remaining_model_calls": None}, 115)
        with self.assertRaises(control.StopRun):
            quota_gate({"ordinary_usage_allowed": True, "remaining_model_calls": 114}, 115)
        quota_gate({"ordinary_usage_allowed": True, "remaining_model_calls": 115}, 115)

    def test_unquantified_quota_exception_is_explicit_and_never_overrides_known_shortfall(self):
        unknown = {"ordinary_usage_allowed": True, "remaining_model_calls": None}
        self.assertEqual(quota_gate(unknown, 115, allow_unquantified=True), "unquantified_explicitly_accepted")
        with self.assertRaises(control.StopRun):
            quota_gate({"ordinary_usage_allowed": True, "remaining_model_calls": 114}, 115,
                       allow_unquantified=True)
        with self.assertRaises(control.StopRun):
            quota_gate({"ordinary_usage_allowed": False, "remaining_model_calls": None}, 115,
                       allow_unquantified=True)

    def test_generation_keeps_all_evaluation_retry_reservations(self):
        budget = Budget(CONFIG, MemoryLog())
        budget.model_normal = 30
        budget.eval_starts = 30
        budget.eval_normal = 30
        budget.eval_cpu_samples = [10.0]
        budget.before_generation()
        budget.eval_starts = 31
        budget.eval_retries = 1
        budget.before_generation()
        budget.eval_retries = 0
        budget.eval_normal = 31
        with self.assertRaises(control.StopRun):
            budget.before_generation()

    def test_generation_rejects_insufficient_empirical_cpu_budget(self):
        budget = Budget(CONFIG, MemoryLog())
        budget.model_normal = budget.eval_starts = budget.eval_normal = 30
        budget.eval_cpu_samples = [100.0]
        with self.assertRaises(control.StopRun):
            budget.before_generation()


class ModelRuntimeTests(unittest.TestCase):
    def test_small_generation_quota_requires_33_calls(self):
        log = MemoryLog()
        backend = Runtime(ROOT, small_config(), log, cli=["fixture-cli"])
        backend.budget.model_normal = backend.budget.eval_starts = backend.budget.eval_normal = 6
        backend.budget.eval_cpu_samples = [1.0]
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": 33}
        with patch("scripts.p0_budget_quota.read_limits", return_value=quota):
            backend.before_generation()
        gate = next(item for item in log.items if item["event"] == "generation_quota_gate")
        self.assertEqual(gate["required_model_calls"], 33)

    def test_generation_quota_gate_rechecks_and_respects_only_explicit_exception(self):
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": None}
        for accepted in (False, True):
            backend = Runtime(ROOT, CONFIG, MemoryLog(), cli=["fixture-cli"],
                              allow_unquantified_quota=accepted)
            backend.budget.model_normal = backend.budget.eval_starts = backend.budget.eval_normal = 30
            backend.budget.eval_cpu_samples = [10.0]
            with patch("scripts.p0_budget_quota.read_limits", return_value=quota):
                if accepted:
                    backend.before_generation()
                else:
                    with self.assertRaises(control.StopRun):
                        backend.before_generation()

    def test_insufficient_evaluation_budget_stops_before_model_call(self):
        backend = Runtime(ROOT, CONFIG, MemoryLog(), cli=["fixture-cli"])
        backend.budget.cpu_limit = 1
        controller = control.Controller(backend, MemoryLog())
        candidate = controller.register("init_01", ())
        with patch("scripts.p0_model_probe.run_command") as runner:
            with self.assertRaises(control.StopRun):
                controller.propose(candidate, "prompt")
        runner.assert_not_called()

    def test_known_network_failure_uses_only_infrastructure_retry_budget(self):
        log = MemoryLog()
        backend = Runtime(ROOT, CONFIG, log, cli=["fixture-cli"])
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1}},
        ]
        replies = [
            {"exit_code": 1, "stdout": "", "stderr": "connection refused"},
            {"exit_code": 0, "stdout": "\n".join(json.dumps(e) for e in events), "stderr": ""},
        ]
        with patch("scripts.p0_model_probe.run_command", side_effect=replies) as runner:
            self.assertEqual(backend.model("fixture", "prompt"), "answer")
        self.assertEqual(runner.call_count, 2)
        self.assertEqual((backend.budget.model_normal, backend.budget.model_retries), (1, 1))
        for call in runner.call_args_list:
            command = call.args[0]
            self.assertNotIn("--ephemeral", command)
            self.assertNotIn("resume", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-luna")
            self.assertIn("model_providers.p0_chatgpt_https.supports_websockets=false", command)
            self.assertIn("model_providers.p0_chatgpt_https.requires_openai_auth=true", command)
        processes = [event["result"] for event in log.items if event["event"] == "model_process"]
        self.assertEqual([process["exit_code"] for process in processes], [1, 0])
        self.assertEqual(processes[0]["stderr"], "connection refused")

    def test_small_batch_network_failure_does_not_start_retry(self):
        log = MemoryLog()
        backend = Runtime(ROOT, small_config(), log, cli=["fixture-cli"])
        result = {"exit_code": 1, "stdout": "", "stderr": "connection refused"}
        with patch("scripts.p0_model_probe.run_command", return_value=result) as runner:
            with self.assertRaises(control.StopRun):
                backend.model("fixture", "prompt")
        self.assertEqual(runner.call_count, 1)
        self.assertEqual((backend.budget.model_normal, backend.budget.model_retries), (1, 0))
        events = [item["event"] for item in log.items]
        self.assertIn("infrastructure_retry_unavailable", events)
        self.assertNotIn("infrastructure_retry", events)

    def test_rate_limit_stops_without_retry_or_reset(self):
        backend = Runtime(ROOT, CONFIG, MemoryLog(), cli=["fixture-cli"])
        result = {"exit_code": 1, "stdout": "", "stderr": "429 rate limit"}
        with patch("scripts.p0_model_probe.run_command", return_value=result) as runner:
            with self.assertRaises(control.StopRun):
                backend.model("fixture", "prompt")
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(backend.budget.model_retries, 0)


@unittest.skipUnless(os.name == "nt", "Windows Job Object 测试")
class WindowsJobTests(unittest.TestCase):
    def test_descendant_cpu_is_counted_after_all_processes_exit(self):
        child = "import time; end=time.monotonic()+0.3\nwhile time.monotonic()<end: pass"
        parent = ("import subprocess,sys; subprocess.run([sys.executable,'-B','-c',"
                  + repr(child) + "], check=True)")
        with WindowsJob() as job:
            process = job.start([sys.executable, "-B", "-c", parent], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 10
            while job.accounting()["active_processes"] and time.monotonic() < deadline:
                time.sleep(0.05)
            process.wait(timeout=5)
            info = job.accounting()
            self.assertEqual(info["active_processes"], 0)
            self.assertGreaterEqual(info["total_processes"], 2)
            self.assertGreater(info["cpu_seconds"], 0.1)

    def test_terminate_removes_live_descendants(self):
        parent = "import subprocess,sys; subprocess.run([sys.executable,'-B','-c','while True: pass'])"
        with WindowsJob() as job:
            process = job.start([sys.executable, "-B", "-c", parent], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 5
            while job.accounting()["total_processes"] < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertGreaterEqual(job.accounting()["total_processes"], 2)
            job.terminate()
            deadline = time.monotonic() + 5
            while job.accounting()["active_processes"] and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(job.accounting()["active_processes"], 0)
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
