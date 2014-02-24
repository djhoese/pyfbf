#!/usr/bin/env python
"""Objects to assist in reading FBF data as a DPL frame-stream.

DPL Reference: https://github.com/rayg-ssec/DplTools
"""
__docformat__ = 'restructuredtext en'

import os
import sys
import logging
import time
from itertools import count

from .slicer import FBFSlicer

LOG = logging.getLogger(__name__)


class fbf2dpl(object):
    """Convert a FBF working directory into a DPL frame stream.

    Can work on a directory that is receiving data in real-time where the FBFs act as circular buffers. For this to
    work properly the `rec_gen` keyword should be used to provide the real-time record numbers. Note that record numbers
    should start at 1 (not 0).

    DPL Reference: https://github.com/rayg-ssec/DplTools
    """
    def __init__(self, work_dir, frame_width=1, buffer_size=0, filename_filter=None, rolling=False, rec_gen=None):
        """Initialize parameters and connect to the working directory for reading.

        :param work_dir: FBF Workspace directory to read data from.
        :param frame_width: Number of values to include in each dictionary generated
        :param buffer_size: Size of FBF file circular buffer (0 means non-circular buffer)
        :param filename_filter: Filter function that returns True if the provided file should be opened for reading.
                                Should return False otherwise.
        :param rolling: Boolean of whether frames should overlap as new records come in (i.e. in a rolling window)
        :param rec_gen: Iterable that yields the record numbers to be read from the file. By default
                        `itertools.count(1)` is used.
        """
        assert(os.path.isdir(work_dir))
        self._work_dir = work_dir
        self._slicer = FBFSlicer(work_dir, buffer_size, filename_filter)
        self._frame_width = frame_width
        self._rec_gen = rec_gen
        self._rolling = rolling

    def close(self):
        return

    def __del__(self):
        try:
            self.close()
        except StandardError:
            LOG.exception("Could not close properly")

    def __call__(self):
        if self._rec_gen is None:
            self._rec_gen = count(1)

        current_frame_width = 0
        for rec in self._rec_gen:
            LOG.debug('processing record %d' % rec)
            current_frame_width += 1

            if not self._rolling and 0 != (current_frame_width % self._frame_width):
                LOG.debug('skipping %d since we are not rolling' % rec)
                continue

            if current_frame_width < self._frame_width:
                LOG.debug('waiting for %d records, only have %d' % (self._frame_width, current_frame_width))
                continue

            first = max(1, rec-self._frame_width+1)
            last = rec

            LOG.debug('grabbing %d thru %d for %d' % (first, last, self._frame_width))
            data = self._slicer(first, last)
            if data is None:
                LOG.warning('???')
                return
            data["first_record"], data["last_record"] = first, last
            LOG.debug('done slicing')
            yield data
            if not self._rolling:
                current_frame_width = 0


def main():
    import argparse
    desc = """Read through a FBF workspace and print data record by record for the specified stems.

    Example:
        python -m ifg.rsh2dpl.fbf /tmp/rsh2fbf airTemp waterTemp
"""
    # see http://docs.python.org/library/argparse.html
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-t', '--test',
                        action="store_true", default=False, help="run self-tests")
    parser.add_argument('--delay', dest='delay', default=1, type=float,
                        help="Delay between record iterations")
    parser.add_argument('--frame_width', dest='frame_width', default=1, type=int,
                        help="Number of elements in the generated frames")
    parser.add_argument('-B', '--buffer-size', dest='buffer_size', default=0, type=int,
                        help='circular buffer size (0 by default)')
    parser.add_argument('-R', '--rolling', default=False, action='store_true', help='rolling records')
    parser.add_argument('-v', '--verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('src', help="source file to process")
    parser.add_argument('stems', nargs='+',
                        help='specify file stems to print out')
    # parser.add_argument('-o', '--output',
    #                 help='location to store output')
    # parser.add_argument('-I', '--include',
    #                 action="append", help="include path to append to GCCXML call")
    args = parser.parse_args()

    if args.test:
        import doctest
        doctest.testmod()
        sys.exit(2)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])
    LOG.debug('verbosity level is %d' % args.verbosity)

    if not args:
        parser.print_help()
        return

    # grab all the low-dimension values
    fnf = lambda fn: fn.split('.')[0] in args.stems

    f2d = fbf2dpl(args.src,
                  frame_width=args.frame_width,
                  buffer_size=args.buffer_size,
                  filename_filter=fnf,
                  rolling=args.rolling)()

    print "Ctrl+C to exit early"
    try:
        for F in f2d:
            print '\n=== record %d~%d' % (F["first_record"], F["last_record"])
            for stem in args.stems:
                print stem
                print F.get(stem, None)
                time.sleep(args.delay)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "Keyboard interrupt caught, exiting early..."

    return 0


if __name__ == '__main__':
    sys.exit(main())
