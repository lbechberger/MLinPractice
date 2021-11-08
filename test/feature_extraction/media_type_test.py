"""
Tests the feature extractor for the media type feature

Created: 04.11.21, 14:12

Author: LDankert
"""
import unittest
import pandas as pd
from src.util import COLUMN_VIDEO, COLUMN_PHOTOS
from src.feature_extraction.media_type import MediaType


class MediaTypeFeatureTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = [COLUMN_PHOTOS,COLUMN_VIDEO]
        self.inputs = [["link","[]","link2","[]", "Shitlink","[]","[]","[]","[]"],[1,1,0,0,1,0,0,1,0]]
        self.expected_outputs = [[1,0,0],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,0,1],[0,1,0],[0,0,1]]
        self.mediatyper = MediaType(self.INPUT_COLUMN)

    def test_mediatype_get_values(self):
        function_output = self.mediatyper._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)

    def test_input_columns(self):
        self.assertEqual(self.mediatyper._input_columns, self.INPUT_COLUMN)

    def test_feature_name(self):
        self.assertEqual(self.mediatyper.get_feature_name(), "media_type")


if __name__ == '__main__':
    unittest.main()
