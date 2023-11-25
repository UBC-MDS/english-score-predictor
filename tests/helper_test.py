import unittest
import helper


class TestHelperMethods(unittest.TestCase):
    # Example test function
    def test_sum(self):
        self.assertEqual(helper.sum(1, 2), 3)
        self.assertEqual(helper.sum(1, -1), 0)
        self.assertEqual(helper.sum(1, 0), 1)
        self.assertEqual(helper.sum(1, 1), 2)


if __name__ == "__main__":
    unittest.main()
