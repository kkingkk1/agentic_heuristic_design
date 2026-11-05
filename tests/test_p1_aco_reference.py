from pathlib import Path
import unittest

import numpy as np

from baselines.aco_reference import bpp_offline_aco, cvrp_aco, mkp_aco, op_aco, tsp_aco
from scripts.p0_controller import check_modification_scope
from scripts.p1_task_contract import load_task_contract
from tests import test_p1_bpp_aco, test_p1_cvrp_aco, test_p1_mkp_aco, test_p1_op_aco, test_p1_tsp_aco


class TestACOReferences(unittest.TestCase):
    def test_references_satisfy_each_white_box_signature(self):
        for module in (tsp_aco, cvrp_aco, op_aco, mkp_aco, bpp_offline_aco):
            name = module.__name__.split(".")[-1]
            code = Path(module.__file__).read_text(encoding="utf-8")
            contract = load_task_contract(name)
            parsed = contract.parse_proposal(f"<thought>reference</thought><code>{code}</code>")
            check_modification_scope(parsed["code"])

    def test_reference_outputs_pass_task_interfaces(self):
        distance = np.array([[1.0, 2.0], [2.0, 1.0]])
        points = np.array([[0.5, 0.5], [0.0, 0.0]])
        demand = np.array([0.0, 5.0])
        prize = np.array([0.1, 1.0])
        weight = np.array([[0.2, 0.1], [0.1, 0.2]])
        self.assertEqual(test_p1_tsp_aco.task.checked_heuristic(tsp_aco.heuristics_v2, distance).shape, (2, 2))
        self.assertEqual(test_p1_cvrp_aco.task.checked_heuristic(cvrp_aco.heuristics_v2, distance, points, demand).shape, (2, 2))
        self.assertEqual(test_p1_op_aco.task.checked_heuristic(op_aco.heuristics_v2, prize, distance).shape, (2, 2))
        self.assertEqual(test_p1_mkp_aco.task.checked_heuristic(mkp_aco.heuristics_v2, prize, weight).shape, (2,))
        self.assertEqual(test_p1_bpp_aco.task.checked_heuristic(bpp_offline_aco.heuristics_v2, np.array([20, 50])).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
