#!/usr/bin/env python
"""Tests for the slicer.py module, specifically the FBFSlicer class.
"""

from pyfbf.slicer import FBFSlicer
from pyfbf.memfbf import dtype_from_path
import mock
import numpy

import os
import sys
import logging
import unittest

FAKE_FBF_FNS = [
    "fake1.real4",
    "fake2.real4.5",
    "fake3.int4",
    "fake4.int4.10"
]


class FakeFilenameFilter(object):
    def __init__(self, allowed=FAKE_FBF_FNS):
        self.allowed = FAKE_FBF_FNS

    def __call__(self, fn):
        return fn in self.allowed


class TestSlicer(unittest.TestCase):
    def _create_fake_fbfs(self, num_records=1, fbfs=FAKE_FBF_FNS):
        for fn in fbfs:
            dtype = dtype_from_path(fn)
            fbf = (100 * numpy.random.rand(num_records, *dtype.shape)).astype(dtype.base)
            fbf.tofile(fn)

    def _remove_fake_fbfs(self, fbfs=FAKE_FBF_FNS):
        for fn in fbfs:
            os.remove(fn)

    @mock.patch("pyfbf.slicer.glob", lambda pat: [])
    def test_empty_dir(self):
        slicer = FBFSlicer(".")
        frame = slicer(1, 15)
        self.assertDictEqual(frame, {})

    def test_angry_filter(self):
        self._create_fake_fbfs(num_records=15)

        try:
            slicer = FBFSlicer(".", filename_filter=lambda fn: False)
            frame = slicer(1, 15)
        finally:
            self._remove_fake_fbfs()

        self.assertDictEqual(frame, {})

    def test_bad_filter(self):
        def fake_filter(fn):
            raise ValueError("Fake Exception")
        slicer = FBFSlicer(".", filename_filter=fake_filter)
        self.assertRaises(ValueError, slicer, (1, 15))

    def test_first_record_last_record_multiple(self):
        self._create_fake_fbfs(num_records=15)

        try:
            slicer = FBFSlicer(".", filename_filter=FakeFilenameFilter())
            frame = slicer(1, 15)
        finally:
            self._remove_fake_fbfs()

        self.assertItemsEqual(frame.keys(), ["fake1", "fake2", "fake3", "fake4"])
        self.assertEqual(frame["fake1"].shape, (15,))
        self.assertEqual(frame["fake2"].shape, (15, 5))
        self.assertEqual(frame["fake3"].shape, (15,))
        self.assertEqual(frame["fake4"].shape, (15, 10))

    def test_first_record_last_record_single(self):
        self._create_fake_fbfs(num_records=1)

        try:
            slicer = FBFSlicer(".", filename_filter=FakeFilenameFilter())
            # should only return what is available
            frame = slicer(1, 15)
        finally:
            self._remove_fake_fbfs()

        self.assertItemsEqual(frame.keys(), ["fake1", "fake2", "fake3", "fake4"])
        self.assertEqual(frame["fake1"].shape, (1,))
        self.assertEqual(frame["fake2"].shape, (1, 5))
        self.assertEqual(frame["fake3"].shape, (1,))
        self.assertEqual(frame["fake4"].shape, (1, 10))

    def test_circular_buffer_simple1(self):
        fbf_fn = "fake_cb.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        slicer = FBFSlicer(".", buffer_size=20, filename_filter=lambda fn: fn == fbf_fn)
        # slicer should handle wrapping around the circular buffer, even multiple times
        frame = slicer(38, 45)
        os.remove(fbf_fn)

        self.assertListEqual(frame.keys(), ["fake_cb"])
        #                                             38, 39, 40,41,42,43,44,45
        self.assertListEqual(list(frame["fake_cb"]), [18, 19, 20, 1, 2, 3, 4, 5])


def main():
    logging.basicConfig(level=logging.DEBUG)
    return unittest.main()

if __name__ == "__main__":
    sys.exit(main())

if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.DEBUG)
