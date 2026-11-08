import unittest

from scripts import p0_controller as control, p0_model_probe as probe
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog


class TestTaskContracts(unittest.TestCase):
    def test_eleven_conditions_read_correct_signature_and_direction(self):
        directions = {"tsp_aco": "min", "cvrp_aco": "min", "bpp_offline_aco": "min",
                      "op_aco": "max", "mkp_aco": "max", "tsp_gls": "min"}
        self.assertEqual(TASKS, directions.keys())
        for name, direction in directions.items():
            for black_box in ((False,) if name == "tsp_gls" else (False, True)):
                with self.subTest(name=name, black_box=black_box):
                    contract = load_task_contract(name, black_box=black_box)
                    self.assertEqual(contract.direction, direction)
                    self.assertEqual(contract.black_box, black_box)
                    self.assertIn(contract.signature, contract.task_prompt())
                    self.assertIn(contract.signature, contract.output_prompt())
                    self.assertNotIn("heuristics_v1", contract.task_prompt())
                    self.assertNotIn("return np.ones", contract.task_prompt())
                    code = "import numpy as np\n" + contract.signature + "\n    return np.ones(1)\n"
                    self.assertEqual(contract.parse_proposal(f"<thought>idea</thought><code>{code}</code>")["code"], code.strip())
                    self.assertEqual(contract.parse_proposal(f"<thought>idea</thought>\n{code}")["code"], code.strip())

    def test_wrong_names_types_and_non_module_output_rejected(self):
        contract = load_task_contract("op_aco", black_box=True)
        wrong_name = contract.signature.replace("node_constraint", "edge_constraint")
        wrong_type = contract.signature.replace(": float", ": int")
        for signature in (wrong_name, wrong_type):
            code = "import numpy as np\n" + signature + "\n    return np.ones(1)"
            with self.subTest(signature=signature), self.assertRaises(probe.ProbeError):
                contract.parse_proposal(f"<thought>idea</thought><code>{code}</code>")
        with self.assertRaises(probe.ProbeError):
            contract.parse_proposal("<thought>idea</thought><code>pass</code>")

    def test_unknown_condition_and_missing_return_rejected(self):
        with self.assertRaises(ValueError):
            load_task_contract("unknown")
        with self.assertRaises(ValueError):
            load_task_contract("tsp_aco", black_box="false")
        contract = load_task_contract("tsp_aco")
        code = "import numpy as np\n" + contract.signature + "\n    pass"
        with self.assertRaises(probe.ProbeError):
            contract.parse_proposal(f"<thought>idea</thought><code>{code}</code>")

    def test_untagged_module_still_requires_thought_and_valid_code(self):
        contract = load_task_contract("tsp_aco")
        code = "import numpy as np\n" + contract.signature + "\n    return np.ones(1)"
        invalid = (code, f"<thought> </thought>\n{code}", f"<thought>idea</thought>\n```python\n{code}\n```",
                   f"<thought>idea</thought>\n<code>{code}", "<thought>idea</thought>\nthis is not Python")
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(probe.ProbeError):
                contract.parse_proposal(text)

    def test_controller_passes_unwrapped_module_without_extra_model_call(self):
        contract = load_task_contract("tsp_aco")
        code = "import numpy as np\n" + contract.signature + "\n    return np.ones_like(distance_matrix)"

        class Backend(FixtureBackend):
            def model(self, stage, prompt):
                self.calls.append((stage, prompt))
                return f"<thought>Use uniform scores.</thought>\n{code}"

        backend, log = Backend(), MemoryLog()
        candidate = control.Controller(backend, log, task_contract=contract).batch([("init_01", (), "fixture")])[0]
        self.assertTrue(candidate.valid)
        self.assertEqual(candidate.code, code)
        self.assertEqual((len(backend.calls), len(backend.evaluations)), (1, 1))

    def test_task_descriptions_name_the_actual_interface_parameters(self):
        op = load_task_contract("op_aco")
        self.assertIn("`maxlen`", op.function_description)
        self.assertNotIn("`max_len`", op.function_description)

        bpp_black = load_task_contract("bpp_offline_aco", black_box=True)
        for parameter in ("node_attr", "node_constraint"):
            with self.subTest(parameter=parameter):
                self.assertIn(f"`{parameter}`", bpp_black.function_description)
        self.assertNotIn("`item_attr`", bpp_black.function_description)

    def test_controller_runs_one_mock_generation_for_each_condition(self):
        for name in sorted(TASKS):
            for black_box in ((False,) if name == "tsp_gls" else (False, True)):
                contract = load_task_contract(name, black_box=black_box)

                class Backend(FixtureBackend):
                    def model(self, stage, prompt):
                        response = super().model(stage, prompt)
                        return response.replace(
                            "def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:", contract.signature
                        ).replace("np.ones_like(distance_matrix)", "np.array([1.0])")

                    def evaluate(self, candidate):
                        result = super().evaluate(candidate)
                        if contract.direction == "max":
                            result["fitness"] = result["average"]
                        return result

                with self.subTest(task=name, black_box=black_box):
                    backend, log = Backend(), MemoryLog()
                    controller = control.Controller(backend, log, task_contract=contract)
                    selected = controller.run()
                    self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (95, 88, 10))
                    calls = dict(backend.calls)
                    self.assertIn(contract.signature, calls["init_01"])
                    self.assertNotIn("return np.", calls["init_01"])
                    expected = "Higher" if contract.direction == "max" else "Lower"
                    self.assertIn(f"A {expected.lower()} objective score is better", calls["critic_initial"])
                    self.assertIn(f"A {expected.lower()} objective is better", calls["critic_1"])
                    self.assertIn(f"A {expected.lower()} objective score is better", calls["fusion_1"])
                    self.assertTrue(all(c.fitness == (c.objective if contract.direction == "max" else -c.objective)
                                        for c in selected))

    def test_two_generations_keep_population_history_and_distinct_provenance(self):
        backend, log = FixtureBackend(), MemoryLog()
        controller = control.Controller(backend, log, generations=2)
        selected = controller.run()
        self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (160, 146, 10))
        self.assertEqual(backend.generation_checks, 2)
        selections = [event for event in log.items if event["event"] == "selection"]
        self.assertEqual([event["generation"] for event in selections], [1, 2])
        second_pair = next(event for event in log.items if event["event"] == "elite_pair" and event["generation"] == 2)
        first_selected = {entry["candidate_id"] for entry in selections[0]["selected"]}
        self.assertTrue({second_pair["a"], second_pair["b"]} <= first_selected)
        registered = {event["candidate_id"]: event for event in log.items if event["event"] == "candidate_registered"}
        self.assertTrue(all(registered[candidate]["generation"] == 2 for candidate in selections[1]["pool"]))
        self.assertEqual([len(controller.history[role]) for role in ("solution_quality_agent", "speed_agent", "integrator")], [2, 2, 2])
        mutations = [prompt for stage, prompt in backend.calls if stage == "mutation_A_solution_quality_agent"]
        self.assertEqual(len(mutations), 2)
        self.assertIn("[History reflection as reference]:\n\n", mutations[0])
        self.assertIn("[History reflection as reference]:\nRetain useful changes.", mutations[1])
        history = [event for event in log.items if event["event"] == "history_appended"]
        self.assertEqual([event["generation"] for event in history], [1, 1, 1, 2, 2, 2])

    def test_stopping_before_second_generation_preserves_first_history(self):
        class StoppingBackend(FixtureBackend):
            def before_generation(self):
                super().before_generation()
                if self.generation_checks == 2:
                    raise control.StopRun("预算不足")

        backend, log = StoppingBackend(), MemoryLog()
        controller = control.Controller(backend, log, generations=2)
        with self.assertRaises(control.StopRun):
            controller.run()
        self.assertEqual([len(controller.history[role]) for role in ("solution_quality_agent", "speed_agent", "integrator")], [1, 1, 1])
        self.assertFalse(any(event.get("generation") == 2 for event in log.items
                             if event["event"] == "candidate_registered"))
        self.assertEqual(log.items[-1]["status"], "stopped")

    def test_six_generation_mock_has_documented_slot_counts(self):
        backend, log = FixtureBackend(), MemoryLog()
        controller = control.Controller(backend, log, generations=6)
        selected = controller.run()
        self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (420, 378, 10))
        self.assertEqual(backend.generation_checks, 6)
        self.assertEqual([event["generation"] for event in log.items if event["event"] == "selection"],
                         list(range(1, 7)))
        self.assertEqual([len(controller.history[role]) for role in ("solution_quality_agent", "speed_agent", "integrator")], [6, 6, 6])

    def test_generation_count_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.Controller(FixtureBackend(), MemoryLog(), generations=value)


if __name__ == "__main__":
    unittest.main()
