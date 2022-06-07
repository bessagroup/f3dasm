# -*- coding: utf-8 -*-

from .context import f3dasm

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(f3dasm.hmm())


if __name__ == '__main__':
    unittest.main()
