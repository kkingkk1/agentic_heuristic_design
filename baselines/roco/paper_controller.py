"""RoCo 论文角色的任务适配；不把 Q/E 角色目标标为 RoCo。"""

from scripts.p0_controller import Controller, candidate_text


ROLE_LABELS = {"Q": "Explorer", "E": "Exploiter", "I": "Integrator"}


class RoCoController(Controller):
    """共享经项目约定接入的代际流程，角色目标采用论文探索／开发语义。"""

    def event(self, kind, **fields):
        for key in ("role", "source_role"):
            if fields.get(key) in ROLE_LABELS:
                fields[key] = ROLE_LABELS[fields[key]]
        super().event(kind, **fields)

    def branch_prompt(self, role, candidate, critic):
        if role == "Q":
            instruction = ("Focus on global diversity and long-term potential. Use creative, future-oriented thinking "
                           "to introduce novel directions combined with the critic's feedback.")
        else:
            instruction = ("Refine and improve the heuristic through short-term gains and local optimizations. "
                           "Use conservative, efficiency-driven thinking to enhance performance incrementally.")
        return (f"{self.task}\n{candidate_text(candidate)}Critic's feedback: {critic['advice']}\n"
                f"Reflection: {critic['reflection']}\nBased on the feedback and reflection, propose a new "
                f"algorithm that improves the previous one. {instruction}\n{self.output}")

    def branch_critic_prompt(self, explorer, exploiter):
        if exploiter.fitness > explorer.fitness:
            instruction = ("The current Exploiter candidate has a better objective than Explorer. Identify successful "
                           "changes and recommend how to build on them.")
            compared = f"Better algorithm (Exploiter):\n{candidate_text(exploiter)}Worse algorithm (Explorer):\n{candidate_text(explorer)}"
        elif exploiter.fitness < explorer.fitness:
            instruction = ("The current Exploiter candidate has a worse objective than Explorer. Identify "
                           "counterproductive choices and suggest how to correct the direction.")
            compared = f"Better algorithm (Explorer):\n{candidate_text(explorer)}Worse algorithm (Exploiter):\n{candidate_text(exploiter)}"
        else:
            instruction = "The objectives are tied. Compare the designs without claiming either score is better."
            compared = f"Explorer algorithm:\n{candidate_text(explorer)}Exploiter algorithm:\n{candidate_text(exploiter)}"
        better = "lower" if self.direction == "min" else "higher"
        return (f"{self.task}\nCompare these evaluated algorithms; a {better} objective is better.\n"
                f"{compared}{instruction} Reflect on the differences and give actionable improvements. "
                "Return exactly <ref>reflection</ref><ans>advice, at most 40 words</ans>.")

    def round_fusion_prompt(self, proposals, better):
        return (f"{self.task}\nExplorer candidate:\n{candidate_text(proposals[0])}"
                f"Exploiter candidate:\n{candidate_text(proposals[1])}"
                "Act as a strategic integrator. Weigh the long-term innovation of Explorer and the "
                f"short-term refinement of Exploiter. A {better.lower()} objective is better. Propose the next "
                "step that balances exploration and exploitation by reasoning through "
                f"their complementary strengths.\n{self.output}")

    def final_fusion_prompt(self, best, reflections):
        return (f"{self.task}\nBest Explorer:\n{candidate_text(best[0], thought=False)}"
                f"Best Exploiter:\n{candidate_text(best[1], thought=False)}"
                f"Reflection summary: {' | '.join(reflections)}\nAct as a strategic integrator. "
                "Consider the long-term innovation potential of the best Explorer, the short-term "
                "refinement capabilities of the best Exploiter, and the reflection insights. Combine "
                "their strongest elements and address those insights, balancing exploration "
                f"and exploitation.\n{self.output}")

    def long_reflection_prompt(self, role, reflections, changes):
        description = ("global diversity and long-term potential" if role == "Q" else
                       "short-term gains and local optimization")
        return (f"{self.task}\nRole: {ROLE_LABELS[role]}, focus on {description}.\n"
                f"Short-term reflections: {' | '.join(reflections)}\n"
                f"Objective changes: {changes[role]}\n"
                "Give constructive hints in fewer than 50 words. Return exactly <ans>...</ans>.")

    def merge_reflections_prompt(self, lessons):
        return ("Merge these two reflections into one concise experience. Return plain text only.\n"
                f"Explorer: {lessons['Q']}\nExploiter: {lessons['E']}")

    def run(self):
        self.event("role_mapping", internal={"Q": "Explorer", "E": "Exploiter", "I": "Integrator"})
        return super().run()
