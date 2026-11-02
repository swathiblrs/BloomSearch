
import unittest
from bloom_search.rewrite import rewrite_query


class FakeHostedClient:
    def chat_json(self, *, system, user, temperature=0):
        return {
            "rewritten_query": "When was Alan Turing born?"
        }


class TestHostedRewrite(unittest.TestCase):

    # Test 1: Verify hosted API rewriting works
    def test_hosted_rewrite(self):
        result = rewrite_query(
            original_query="When was he born?",
            context=["We are discussing Alan Turing."],
            hosted_client=FakeHostedClient(),
            force_rewrite=True,
            prompt_version="improved",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.rewritten_query,
            "When was Alan Turing born?",
        )

    # Test 2: Reject rewrites that change dates
    def test_rejects_changed_date(self):
        class BadHostedClient:
            def chat_json(self, *, system, user, temperature=0):
                return {
                    "rewritten_query": "What happened in 2021?"
                }

        result = rewrite_query(
            original_query="What happened in 2020?",
            context=[],
            hosted_client=BadHostedClient(),
            force_rewrite=True,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.rewritten_query,
            "What happened in 2020?",
        )

    # Test 3: Reject changes to locations, entities, and constraints
    def test_other_intent_changes(self):
        cases = [
            (
                "Where is Knoxville located?",
                "Where is Nashville located?",
            ),
            (
                "When was Alan Turing born?",
                "When was Albert Einstein born?",
            ),
            (
                "Find hotels under $100 in Knoxville",
                "Find hotels in Knoxville",
            ),
        ]

        for original, rewritten in cases:
            with self.subTest(original=original, rewritten=rewritten):

                class FakeClient:
                    def chat_json(self, *, system, user, temperature=0):
                        return {"rewritten_query": rewritten}

                result = rewrite_query(
                    original_query=original,
                    context=[],
                    hosted_client=FakeClient(),
                    force_rewrite=True,
                )

                self.assertFalse(
                    result.accepted,
                    f"Unsafe rewrite was accepted: {rewritten}",
                )
                self.assertEqual(
                    result.rewritten_query,
                    original,
                )


if __name__ == "__main__":
    unittest.main()
