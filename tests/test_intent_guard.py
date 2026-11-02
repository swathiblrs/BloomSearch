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


if __name__ == "__main__":
    unittest.main()