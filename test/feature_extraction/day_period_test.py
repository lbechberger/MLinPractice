"""
test the class of the feature extraction for the day period

Created: 04.11.21, 13:58

Author: LDankert
"""
import unittest
from src.feature_extraction.day_period import DayPeriod


class DayPeriodTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "dummy name"
        self.inputs = [["Morning","Evening","Night","Afternoon","Afternoon","Night"]]
        self.error_inputs =[["No Time"],["asda s"], [10], ["Evening","Night","No Time"]]
        self.expected_outputs = [[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[0,0,1,0],[1,0,0,0]]
        self.dayperioder = DayPeriod(self.INPUT_COLUMN)

    def test_dayperiod_get_values(self):
        function_output = self.dayperioder._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)

    def test_dayperioder_get_values_exception(self):
        for input in self.error_inputs:
            self.assertRaises(Exception, self.dayperioder._get_values, input)

    def test_input_columns(self):
        self.assertEqual(self.dayperioder._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.dayperioder.get_feature_name(), self.INPUT_COLUMN)


if __name__ == '__main__':
    unittest.main()
