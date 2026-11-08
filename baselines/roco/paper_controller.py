"""RoCo 论文角色的任务适配。"""

from scripts.p0_controller import Controller


ROLE_LABELS = {"explorer": "Explorer", "exploiter": "Exploiter", "integrator": "Integrator"}


class RoCoController(Controller):
    """共享经项目约定接入的代际流程，角色目标采用论文探索／开发语义。"""

    design_roles = ("explorer", "exploiter")
    role_details = {
        "explorer": {
            "label": "Explorer", "focus": "global diversity and long-term potential", "identity": "",
            "opening": "Here is an algorithm proposed by previous agents:",
            "direction": "focusing on {focus}. Use creative, future-oriented thinking to introduce novel directions combined with critic's feedback.",
        },
        "exploiter": {
            "label": "Exploiter", "focus": "short-term gains and local optimizations", "identity": "",
            "opening": "Here are some algorithms proposed by previous agents:",
            "direction": "to refine and improve it through {focus}. Use conservative, efficiency-driven thinking to enhance performance incrementally.",
        },
    }
    fusion_wording = {
        "first_final_label": "explorer", "second_final_label": "exploiter",
        "round_priorities": "the long-term innovation of the explorer, the short-term refinement of the exploiter",
        "balance": "exploration and exploitation",
        "first_final_priority": "long-term innovation potential from the best explorer",
        "second_final_priority": "short-term refinement capabilities from the best exploiter",
    }

    def event(self, kind, **fields):
        for key in ("role", "source_role"):
            if fields.get(key) in ROLE_LABELS:
                fields[key] = ROLE_LABELS[fields[key]]
        super().event(kind, **fields)

    def run(self):
        self.event("role_mapping", internal=ROLE_LABELS.copy())
        return super().run()
