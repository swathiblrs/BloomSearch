import unittest

from bloom_search.intent_guard import preserves_intent


class IntentGuardTests(unittest.TestCase):
    def test_accepts_rewrite_that_keeps_person(self):
        accepted, problems = preserves_intent(
            "What about him?",
            "What is Gary Cherone known for?",
            ["Tell me about Gary Cherone."],
        )
        self.assertTrue(accepted)
        self.assertEqual(problems, [])

    def test_rejects_rewrite_that_changes_person(self):
        accepted, problems = preserves_intent(
            "What about him?",
            "What is David Lee Roth known for?",
            ["Tell me about Gary Cherone."],
        )
        self.assertFalse(accepted)
        self.assertTrue(problems)

    def test_rejects_rewrite_that_drops_topic_terms(self):
        accepted, problems = preserves_intent(
            "What moves was he most known for?",
            "What was Pelé most known for?",
            ["What position did Pelé play?"],
        )
        self.assertFalse(accepted)
        self.assertTrue(any("important query terms" in problem for problem in problems))

    def test_rejects_rewrite_that_removes_continuation_meaning(self):
        accepted, problems = preserves_intent(
            "What else did he do?",
            "What was the name of the former NHL lockout?",
            ["What happened to Dominik Hašek in 2005?"],
        )
        self.assertFalse(accepted)
        self.assertTrue(any("continuation meaning" in problem for problem in problems))

    def test_rejects_subject_object_reversal(self):
        accepted, problems = preserves_intent(
            "Who was he accused of assaulting?",
            "Who was accused of assaulting Jack White?",
            ["What controversy surrounds Jack White?"],
        )
        self.assertFalse(accepted)
        self.assertTrue(any("reversed" in problem for problem in problems))

    def test_rejects_rewrite_that_removes_original_constraints(self):
        accepted, problems = preserves_intent(
            "Did he date anyone famous before his first marriage?",
            "Did Stephen Stills and Veronique Sanson have any children?",
            ["When did Stephen Stills and Veronique Sanson divorce?"],
        )
        self.assertFalse(accepted)
        self.assertTrue(any("restrictions removed" in problem for problem in problems))

    def test_accepts_entity_resolution_that_preserves_topic(self):
        accepted, problems = preserves_intent(
            "Why did he design the bags?",
            "Why did Tommy Hilfiger design the bags?",
            ["Tommy Hilfiger designed limited-edition handbags."],
        )
        self.assertTrue(accepted)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
