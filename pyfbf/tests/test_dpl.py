#!/usr/bin/env python
"""Tests for the dpl.py module, specifically the fbf2dpl class.
"""

from pyfbf.dpl import fbf2dpl
from pyfbf.memfbf import dtype_from_path
import mock
import numpy

import os
import sys
import logging
import unittest


class TestFBF2DPL(unittest.TestCase):
    def test_static_width_1(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        f2d = fbf2dpl('.', filename_filter=lambda fn: fn == fbf_fn)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        for item in all_frames:
            self.assertEqual(item.keys(), ["fake1"])
        self.assertEqual(len(all_frames), 20)
        self.assertListEqual([x["fake1"][0] for x in all_frames], list(fbf_arr))

        os.remove(fbf_fn)

    def test_static_width_5(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        f2d = fbf2dpl('.', frame_width=5, filename_filter=lambda fn: fn == fbf_fn)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        for item in all_frames:
            self.assertEqual(item.keys(), ["fake1"])
        self.assertEqual(len(all_frames), 4)
        self.assertListEqual([list(x["fake1"]) for x in all_frames],
                             [range(1, 6), range(6, 11), range(11, 16), range(16, 21)])

        os.remove(fbf_fn)

    def test_static_width_5_small_file(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 4, dtype=numpy.float32)  # 1 - 3
        fbf_arr.tofile(fbf_fn)

        f2d = fbf2dpl('.', frame_width=5, filename_filter=lambda fn: fn == fbf_fn)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        # We couldn't get an entire frame so we returned nothing
        self.assertListEqual(all_frames, [])

        os.remove(fbf_fn)

    def test_static_width_1_rolling(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        f2d = fbf2dpl('.', filename_filter=lambda fn: fn == fbf_fn, rolling=True)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        for item in all_frames:
            self.assertEqual(item.keys(), ["fake1"])
        self.assertEqual(len(all_frames), 20)
        self.assertListEqual([x["fake1"][0] for x in all_frames], list(fbf_arr))

        os.remove(fbf_fn)

    def test_static_width_5_rolling(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        f2d = fbf2dpl('.', frame_width=5, filename_filter=lambda fn: fn == fbf_fn, rolling=True)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        for item in all_frames:
            self.assertEqual(item.keys(), ["fake1"])
        self.assertEqual(len(all_frames), 16)
        for idx, item in enumerate(all_frames):
            self.assertListEqual(list(item["fake1"]), range(idx+1, idx+6))

        os.remove(fbf_fn)

    def test_buffer_width_5_rec_gen(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        rec_gen = range(18, 38)
        f2d = fbf2dpl('.', frame_width=5, buffer_size=20, filename_filter=lambda fn: fn == fbf_fn, rec_gen=rec_gen)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        for item in all_frames:
            self.assertEqual(item.keys(), ["fake1"])
        self.assertEqual(len(all_frames), 4)
        self.assertListEqual([list(x["fake1"]) for x in all_frames],
                             [[18, 19, 20, 1, 2],
                              [3, 4, 5, 6, 7],
                              [8, 9, 10, 11, 12],
                              [13, 14, 15, 16, 17]])

        os.remove(fbf_fn)

    def test_buffer_width_5_rolling(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        rec_gen = range(18, 38)
        f2d = fbf2dpl('.', frame_width=5, buffer_size=20, filename_filter=lambda fn: fn == fbf_fn, rolling=True,
                      rec_gen=rec_gen)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        for item in all_frames:
            self.assertEqual(item.keys(), ["fake1"])
        self.assertEqual(len(all_frames), 16)
        check_list = range(1, 21) + range(1, 21) + range(1, 21)  # 'simulate' the circular buffer
        for idx, item in enumerate(all_frames):
            # Subtract 1 because record 18 is index 17
            # Add 5 because we want a total of 5 elements in each frame
            self.assertListEqual(list(item["fake1"]), check_list[idx+18-1:idx+18-1+5])

        os.remove(fbf_fn)

    def test_rec_gen_skips(self):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        rec_gen = [18, 25, 37]
        f2d = fbf2dpl('.', frame_width=5, buffer_size=20, filename_filter=lambda fn: fn == fbf_fn, rec_gen=rec_gen)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        self.assertListEqual(all_frames, [])
        # for item in all_frames:
        #     self.assertEqual(item.keys(), ["fake1"])
        # self.assertEqual(len(all_frames), 3)
        # self.assertListEqual([list(x["fake1"]) for x in all_frames],
        #                      [[18, 19, 20, 1, 2],
        #                       [5, 6, 7, 8, 9],
        #                       [17, 18, 19, 20, 21]])

        os.remove(fbf_fn)

    def test_rec_gen_skips_rolling(self):
        """Test that we don't return any frames when the record generator is not incrementing by 1 until we have enough
        records.
        """
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.arange(1, 21, dtype=numpy.float32)  # 1 - 20
        fbf_arr.tofile(fbf_fn)

        rec_gen = [18, 25, 37, 41, 43]
        f2d = fbf2dpl('.', frame_width=5, buffer_size=20, filename_filter=lambda fn: fn == fbf_fn,
                      rec_gen=rec_gen, rolling=True)
        frame_gen = f2d()
        all_frames = list(frame_gen)

        self.assertEqual(len(all_frames), 1)
        self.assertListEqual(list(all_frames[0]["fake1"]), [19, 20, 1, 2, 3])

        os.remove(fbf_fn)

def main():
    logging.basicConfig(level=logging.DEBUG)
    return unittest.main()

if __name__ == "__main__":
    sys.exit(main())

if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.DEBUG)
