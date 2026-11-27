import unittest

from research.human_study.analyze_responses import analyze, macro_f1
from research.human_study.prepare_items import (
    parse_interpretations,
    targeted_clarification,
)
from research.human_study.prepare_participant_packet import packet_rows


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

    def test_targeted_clarification_uses_recorded_interpretations(self):
        interpretations = parse_interpretations(
            "['Which political party took control?', 'Which individual leader took control?']"
        )
        question = targeted_clarification(interpretations)
        self.assertIn("political party", question)
        self.assertIn("individual leader", question)
        self.assertTrue(question.startswith("Do you mean"))
        self.assertNotIn("what you want to know about", question)

    def test_participant_packet_hides_system_action_and_uses_opaque_ids(self):
        key = [
            {
                "item_id": "Q01",
                "original_query": "Example?",
                "context": "",
                "system_action": "clarify",
                "proposed_rewrite": "",
                "clarification_question": "Which interpretation?",
            }
        ]
        actions, outputs, responses = packet_rows(key, "P01", seed=42)
        self.assertNotIn("system_action", actions[0])
        self.assertNotIn("system_action", outputs[0])
        self.assertNotIn("system_action", responses[0])
        self.assertNotIn("clarification_question", actions[0])
        self.assertIn("clarification_question", outputs[0])
        self.assertEqual(actions[0]["item_id"], "Q01")
        self.assertEqual(responses[0]["participant_id"], "P01")

    def test_analysis_joins_hidden_key_and_reports_all_task_success(self):
        key = [
            {"item_id": "Q01", "system_action": "keep"},
            {"item_id": "Q02", "system_action": "rewrite"},
            {"item_id": "Q03", "system_action": "clarify"},
        ]
        responses = [
            {
                "participant_id": "P01",
                "item_id": "Q01",
                "correct_action": "keep",
                "rewrite_preserves_intent": "",
                "clarification_usefulness_1_to_5": "",
                "task_success_0_or_1": "1",
                "trust_1_to_5": "5",
            },
            {
                "participant_id": "P01",
                "item_id": "Q02",
                "correct_action": "rewrite",
                "rewrite_preserves_intent": "yes",
                "clarification_usefulness_1_to_5": "",
                "task_success_0_or_1": "1",
                "trust_1_to_5": "4",
            },
            {
                "participant_id": "P01",
                "item_id": "Q03",
                "correct_action": "clarify",
                "rewrite_preserves_intent": "",
                "clarification_usefulness_1_to_5": "4",
                "task_success_0_or_1": "0",
                "trust_1_to_5": "3",
            },
        ]
        result = analyze(responses, key)
        self.assertEqual(result["action_policy"]["macro_f1"], 1.0)
        self.assertEqual(result["perceived_task_success"]["overall"]["evaluated"], 3)
        self.assertEqual(
            result["clarification"]["primary_perceived_task_success"]["success_rate"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
