import unittest

from calculator import calculate_percentage_change


class PercentageChangeTests(unittest.TestCase):
    def test_increase(self):
        self.assertAlmostEqual(calculate_percentage_change(100, 150), 50.0)

    def test_decrease(self):
        self.assertAlmostEqual(calculate_percentage_change(80, 60), -25.0)

    def test_zero_original_raises(self):
        with self.assertRaises(ValueError):
            calculate_percentage_change(0, 10)


if __name__ == "__main__":
    unittest.main()
