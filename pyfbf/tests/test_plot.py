#!/usr/bin/env python
# encoding: utf-8

from pyfbf.plot import plot_binary_file
import unittest
import numpy
import os
import sys
import logging

try:
    # python 3 includes mock
    from unittest import mock
except ImportError:
    import mock

LOG = logging.getLogger(__name__)


class TestBinaryPlot(unittest.TestCase):
    @mock.patch("pyfbf.plot.plt")
    def test_plot_1d(self, plt):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
        fbf_arr.tofile(fbf_fn)

        try:
            plot_fn = plot_binary_file(fbf_fn)
        finally:
            os.remove(fbf_fn)

        plt.savefig.assert_called_with(plot_fn)
        self.assertTrue(plt.plot.called)
        self.assertEqual(plt.plot.call_count, 1)

    @mock.patch("pyfbf.plot.plt")
    def test_plot_1d_dpi(self, plt):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
        fbf_arr.tofile(fbf_fn)

        try:
            plot_fn = plot_binary_file(fbf_fn, dpi=500)
        finally:
            os.remove(fbf_fn)

        plt.savefig.assert_called_with(plot_fn, dpi=500)
        self.assertTrue(plt.plot.called)
        self.assertEqual(plt.plot.call_count, 1)

    @mock.patch("pyfbf.plot.plt")
    def test_plot_1d_fill(self, plt):
        fbf_fn = "fake1.real4"
        fbf_arr = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
        fbf_arr.tofile(fbf_fn)

        try:
            plot_fn = plot_binary_file(fbf_fn, fill_value=2)
        finally:
            os.remove(fbf_fn)

        plt.savefig.assert_called_with(plot_fn)
        self.assertTrue(plt.plot.called)
        self.assertEqual(plt.plot.call_count, 1)

    @mock.patch("pyfbf.plot.plt")
    def test_plot_2d(self, plt):
        fbf_fn = "fake1.real4.5"
        fbf_arr = numpy.array(range(25), dtype=numpy.float32).reshape((5, 5))
        fbf_arr.tofile(fbf_fn)

        try:
            plot_fn = plot_binary_file(fbf_fn)
        finally:
            os.remove(fbf_fn)

        plt.savefig.assert_called_with(plot_fn)
        self.assertTrue(plt.imshow.called)
        self.assertEqual(plt.imshow.call_count, 1)
        self.assertTrue(plt.colorbar.called)

    @mock.patch("pyfbf.plot.plt")
    def test_plot_2d_dpi(self, plt):
        fbf_fn = "fake1.real4.5"
        fbf_arr = numpy.array(range(25), dtype=numpy.float32).reshape((5, 5))
        fbf_arr.tofile(fbf_fn)

        try:
            plot_fn = plot_binary_file(fbf_fn, dpi=500)
        finally:
            os.remove(fbf_fn)

        plt.savefig.assert_called_with(plot_fn, dpi=500)
        self.assertTrue(plt.imshow.called)
        self.assertEqual(plt.imshow.call_count, 1)
        self.assertTrue(plt.colorbar.called)

    @mock.patch("pyfbf.plot.plt")
    def test_plot_2d_fill(self, plt):
        fbf_fn = "fake1.real4.5"
        fbf_arr = numpy.array(range(25), dtype=numpy.float32).reshape((5, 5))
        fbf_arr.tofile(fbf_fn)

        try:
            plot_fn = plot_binary_file(fbf_fn, fill_value=2)
        finally:
            os.remove(fbf_fn)

        plt.savefig.assert_called_with(plot_fn)
        self.assertTrue(plt.imshow.called)
        self.assertEqual(plt.imshow.call_count, 1)
        self.assertTrue(plt.colorbar.called)


def main():
    logging.basicConfig(level=logging.DEBUG)
    return unittest.main()

if __name__ == "__main__":
    sys.exit(main())

if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.DEBUG)
