import unittest

from research.human_study.analyze_responses import macro_f1


class HumanStudyMetricTests(unittest.TestCase):
    def test_macro_f1_is_one_for_perfect_three_class_predictions(self):
        labels = ["keep", "rewrite", "clarify"]
        score, per_action = macro_f1(labels, labels)
        self.assertEqual(score, 1.0)
        self.assertEqual(per_action, {"keep": 1.0, "rewrite": 1.0, "clarify": 1.0})

    def test_macro_f1_penalizes_wrong_actions(self):
        score, _ = macro_f1(
            ["keep", "rewrite", "clarify"],
            ["rewrite", "rewrite", "keep"],
        )
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
