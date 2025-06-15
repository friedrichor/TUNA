from collections import defaultdict


class CaptioningScorer:
    def __init__(self):
        self.skill_types = ["camera", "scene", "action", "attribute"]
        self.score_types = ["global"] + self.skill_types
        self.categories = ["All", "Low-Dynamic", "High-Dynamic", "Multi-Scene", "Multi-Subject"]

        self.category_scores = {
            cat: {
                t: {"P": [], "R": [], "F1": []}
                for t in self.score_types
            } for cat in self.categories
        }

    def calculate_scores_for_type(self, counter):
        if counter["weight_sum"] == 0:
            return {"P": None, "R": None, "F1": None}

        P = counter["entailment"] / counter["entailment_or_contradiction"] if counter["entailment_or_contradiction"] > 0 else 0
        R = counter["entailment"] / counter["weight_sum"]
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

        return {"P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3)}

    def score_instance(self, events_gt, relationship):
        counter = {
            t: {"entailment": 0, "entailment_or_contradiction": 0, "weight_sum": 0}
            for t in self.score_types
        }

        for ref_event, eval_event in zip(events_gt, relationship):
            try:
                ve_ref = ref_event["visual_elements"]
                ve_eval = eval_event["visual_elements"]
            except:
                continue

            for gt_el, pred_el in zip(ve_ref, ve_eval):
                rel = pred_el.get("relationship", "lack")
                typ = gt_el["type"]
                w = gt_el["weight"]

                if rel == "entailment":
                    counter["global"]["entailment"] += w
                    counter[typ]["entailment"] += w
                if rel in ["entailment", "contradiction"]:
                    counter["global"]["entailment_or_contradiction"] += w
                    counter[typ]["entailment_or_contradiction"] += w
                counter["global"]["weight_sum"] += w
                counter[typ]["weight_sum"] += w

        return {t: self.calculate_scores_for_type(counter[t]) for t in self.score_types}

    def update_totals(self, score, categories):
        for cat in ["All"] + categories:
            for t in self.score_types:
                if score[t]["F1"] is not None:
                    self.category_scores[cat][t]["P"].append(score[t]["P"])
                    self.category_scores[cat][t]["R"].append(score[t]["R"])
                    self.category_scores[cat][t]["F1"].append(score[t]["F1"])

    def compute_averages(self):
        avg = {}
        for cat in self.categories:
            avg[cat] = {}
            for t in self.score_types:
                p_vals = self.category_scores[cat][t]["P"]
                r_vals = self.category_scores[cat][t]["R"]
                f1_vals = self.category_scores[cat][t]["F1"]
                if len(f1_vals) == 0:
                    avg[cat][t] = {"P": None, "R": None, "F1": None}
                else:
                    avg[cat][t] = {
                        "P": round(sum(p_vals) / len(p_vals), 3),
                        "R": round(sum(r_vals) / len(r_vals), 3),
                        "F1": round(sum(f1_vals) / len(f1_vals), 3),
                    }
        return avg
