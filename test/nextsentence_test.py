#!/usr/bin/env ptyhon3
"""
This module tests nextsentence dataset preparation
"""
import unittest

import numpy as np

import common.nextsentence as ns

TEST_NWINDOW = 16

def _histogram(results):
    buckets = [0] * TEST_NWINDOW
    for result in results:
        buckets[result] += 1
    return buckets

class NSTest(unittest.TestCase):
    """
    Test Model module
    """

    def test_ns_choose(self):
        """
        Test next sentence choosing
        """
        results = [ns.choose_second_sentence(TEST_NWINDOW, 0) for i in range(100)]
        hist = _histogram(results)
        neighs = hist[1]
        nneighs = np.sum(hist[2:])
        self.assertTrue(abs(neighs - nneighs) < 20)

        results = [ns.choose_second_sentence(TEST_NWINDOW, 15) for i in range(100)]
        hist = _histogram(results)
        neighs = hist[14]
        nneighs = np.sum(hist[:14])
        self.assertTrue(abs(neighs - nneighs) < 20)

        results = [ns.choose_second_sentence(TEST_NWINDOW, 7) for i in range(100)]
        hist = _histogram(results)
        neighs = hist[6] + hist[8]
        nneighs = np.sum(hist[:6]) + np.sum(hist[9:])
        self.assertTrue(abs(neighs - nneighs) < 20)

if __name__ == '__main__':
    unittest.main()
