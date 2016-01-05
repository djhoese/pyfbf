#!/usr/bin/env python
# encoding: utf-8
"""Script and utility functions for easy plotting of FBF files.

The plots created here are for quicklook views of the data in the files and may not suit complex needs.
"""
__docformat__ = "restructuredtext en"

from pyfbf.memfbf import FBF
import logging
import numpy

# Use builtin 'agg' for systems with no graphical backend
import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt


LOG = logging.getLogger(__name__)
DEFAULT_FILE_PATTERN = "result_*.real4.*.*"


def plot_binary_file(bf, fill_value=None, dpi=None, vmin=None, vmax=None):
    data = FBF(bf)
    img_fn = "plot_{}.png".format(data.stemname)

    if fill_value is not None:
        if numpy.isnan(fill_value):
            data = numpy.ma.masked_where(numpy.isfinite(data), data)
        else:
            data = numpy.ma.masked_where(data == fill_value, data)

    if LOG.getEffectiveLevel() <= logging.DEBUG:
        LOG.debug("Data Min: {}; Data Max: {}".format(numpy.min(data), numpy.max(data)))

    plt.figure()

    if data.ndim == 1:
        plt.plot(data)
    else:
        plt.imshow(data, vmin=vmin, vmax=vmax)
        plt.bone()
        plt.colorbar()

    if dpi:
        plt.savefig(img_fn, dpi=dpi)
    else:
        plt.savefig(img_fn)

    plt.close()
    return img_fn


def sci_float(x):
    x = x.replace("\"", "")
    x = x.replace("\'", "")
    return float(str(x))


def main():
    from argparse import ArgumentParser
    description = "Plot flat binary files using matplotlib"
    parser = ArgumentParser(description=description)
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    parser.add_argument("-f", dest="fill_value", default=None, type=sci_float,
                        help="Specify the fill value of the input file(s)")
    # parser.add_argument("-w", dest="workspace", default='.',
    #                     help="Workspace to look for flat binary files")
    parser.add_argument('-d', '--dpi', dest="dpi", default=None, type=float,
                        help="Specify the dpi for the resulting figure, higher dpi will result in larger figures and longer run times")
    parser.add_argument('--vmin', dest="vmin", default=None, type=sci_float,
                        help="Specify minimum colorbar value for 2D plots. Defaults to minimum value of data.")
    parser.add_argument('--vmax', dest="vmax", default=None, type=sci_float,
                        help="Specify maximum brightness value for 2D plots. Defaults to maximum value of data.")
    parser.add_argument("binary_files", nargs="*",
                        help="list of flat binary files to be plotted in the current directory")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    for bf in args.binary_files:
        LOG.info("Plotting '{}'".format(bf))
        try:
            plot_binary_file(bf, fill_value=args.fill_value, dpi=args.dpi, vmin=args.vmin, vmax=args.vmax)
        except StandardError:
            LOG.exception("Could not plot '{}'".format(bf))

if __name__ == "__main__":
    import sys

    sys.exit(main())

