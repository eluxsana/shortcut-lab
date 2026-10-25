"""Numerical and scientific checks, independent of the presentation."""

import math
import unittest

import train


class ExperimentTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        rows = train.dataset(43, 0.8, 3)
        weights = [0.7, -0.2, 0.13]
        _, gradient = train.loss_gradient(rows, weights)
        for i in range(3):
            left, right = weights.copy(), weights.copy()
            left[i] -= 1e-5
            right[i] += 1e-5
            estimate = (
                train.loss_gradient(rows, right)[0] - train.loss_gradient(rows, left)[0]
            ) / 2e-5
            self.assertAlmostEqual(estimate, gradient[i], places=8)

    def test_stable_only_gradient(self):
        _, gradient = train.loss_gradient(train.dataset(30, 0.9, 4), [0, 0, 0], True)
        self.assertEqual(gradient[1], 0)

    def test_common_random_numbers(self):
        low, high = train.dataset(100, 0.1, 2), train.dataset(100, 0.9, 2)
        self.assertEqual([(r[0], r[2]) for r in low], [(r[0], r[2]) for r in high])
        self.assertTrue(all(not a[3] or b[3] for a, b in zip(low, high)))

    def test_deterministic_and_split_seeds(self):
        self.assertEqual(train.dataset(10, 0.5, 17), train.dataset(10, 0.5, 17))
        self.assertNotEqual(train.dataset(10, 0.5, 17), train.dataset(10, 0.5, 9281))

    def test_wilson_reference_and_empty_group(self):
        lo, hi = train.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.236593090512564, places=12)
        self.assertAlmostEqual(hi, 0.763406909487436, places=12)
        self.assertIsNone(train.wilson(0, 0))
        result = train.evaluate(
            {"weights": [1, 0, 0], "stable_only": True}, train.dataset(20, 1, 1)
        )
        self.assertIsNone(result["conflicting"]["accuracy"])

    def test_training_loss_and_shift(self):
        data = train.dataset(800, 0.95, 17)
        erm = train.train(data)
        balanced = train.train(train.dataset(800, 0.5, 17))
        self.assertLess(erm["losses"][-1]["loss"], erm["losses"][0]["loss"])
        shifted = train.dataset(1500, 0.05, 9281)
        self.assertGreater(
            train.evaluate(balanced, shifted)["overall"]["accuracy"],
            train.evaluate(erm, shifted)["overall"]["accuracy"] + 0.25,
        )
        self.assertGreater(erm["weights"][1], balanced["weights"][1] + 0.5)

    def test_extreme_logits_are_finite(self):
        self.assertEqual(train.sigmoid(-1000), 0)
        self.assertEqual(train.sigmoid(1000), 1)
        loss, grad = train.loss_gradient([(1000, 1000, 0, True)], [1000, 1000, 0])
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(all(math.isfinite(g) for g in grad))


if __name__ == "__main__":
    unittest.main()
