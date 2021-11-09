"""
test the class of the feature extraction for the day period

Created: 04.11.21, 13:58

Author: LDankert
"""
import unittest
from src.feature_extraction.day_period import DayPeriod


class DayPeriodTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "day_period"
        self.inputs = [["07:02:23","22:03:45","02:02:02","14:56:01","12:30:10","05:55:55"]]
        self.expected_outputs = [[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[0,0,1,0],[1,0,0,0]]
        self.dayperioder = DayPeriod(self.INPUT_COLUMN)

    def test_dayperiod_get_values(self):
        function_output = self.dayperioder._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)

    def test_input_columns(self):
        self.assertEqual(self.dayperioder._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.dayperioder.get_feature_name(), self.INPUT_COLUMN)


if __name__ == '__main__':
    unittest.main()
