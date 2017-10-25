import unittest

from tdparse.helper import read_config

class TestHelper(unittest.TestCase):
    def test_read_config(self):
        self.assertEqual(read_config('unit_test_dong_data'),
                         './tests/data/dong_sent.txt')
        with self.assertRaises(ValueError):
            read_config('nothing here',
                        msg='nothing here should not be in the config.yaml')
