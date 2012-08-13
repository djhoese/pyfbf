#!/usr/bin/env python


import os
import sys
import logging
import time
from subprocess import Popen as popen
from Queue import Queue, Empty as EmptyException
from glob import glob
from numpy import append

from .slicer import fbf_slicer

LOG = logging.getLogger(__name__)

def _onesies():
    n = 0
    while True:
        n+=1
        yield n


class fbf2dpl(object):
    """convert a FBF non-circular working directory into a DPL frame stream
    """
    _work_dir = None          # data we're observing
    _slicer = None            # fbf_slicer instance
    _eob_sleep = None         # minimum time delay between frames in seconds
    _rec_gen = None           # source for record numbers, _onesies() if nothing else provided
    _frame_width = None       # number of records in a frame
    _rolling = None           # whether frames should overlap as new records come in

    def __init__(self, work_dir, frame_width=1, buffer_size=0, filename_filter=None, rolling=False, eob_sleep=None, rec_gen=None):
        """
        start up fpf2dpl for a given rsh2fbf work directory
        optionally provide a filter predicate for which filenames to open into the frame
        optionally provide a record generator which blocks on data availability events, e.g. ifg.core.udp_events
            if record_generator is True and rsh2fbf.pid is available in work directory, will follow that
        """
        assert(os.path.isdir(work_dir))
        self._work_dir = work_dir
        self._eob_sleep = eob_sleep
        self._slicer = fbf_slicer(work_dir, buffer_size, filename_filter)
        self._frame_width = frame_width
        self._rec_gen = rec_gen
        self._rolling = rolling

    def close(self):
        return

    def _rsh2fbf_event_gen(self):
        "connect to the UDP broadcasts of a running rsh2fbf using its pidfile"
        from ifg.core.udp_events import udp_event_gen
        LOG.debug('looking for rsh2fbf.pid')
        fnpid = os.path.join(self._work_dir, 'rsh2fbf.pid')
        if not os.path.isfile(fnpid):
            LOG.warning('no rsh2fbf.pid found')
            return None
        with file(fnpid,'rt') as fp:
            pid = int(fp.read().strip())
            return udp_event_gen(yield_recs=True, MSG='FBFnewRecords', PID=pid)


    def __call__(self):
        next_time = time.time() if self._eob_sleep else None

        if self._rec_gen is True:
            self._rec_gen = self._rsh2fbf_event_gen()
        if self._rec_gen is None:
            self._rec_gen = _onesies()

        for rec in self._rec_gen:
            LOG.debug('processing record %d' % rec)
            if self._eob_sleep:
                now = time.time()
                delta = next_time - now
                if delta>0.0:
                    time.sleep(delta)
                next_time = now + self._eob_sleep

            if not self._rolling and 0 != (rec % self._frame_width):
                LOG.debug('skipping %d since we are not rolling' % rec)
                continue

            if rec < self._frame_width:
                LOG.debug('waiting for %d records' % self._frame_width)
                continue

            first = max(1,rec-self._frame_width+1)
            last = rec

            LOG.debug('grabbing %d thru %d for %d' % (first,last,self._frame_width))
            data = self._slicer(first, last)
            if data is None:
                LOG.warning('???')
                return
            data.first_record, data.last_record = first, last
            LOG.debug('done slicing')
            yield data


def main():
    import argparse
    desc = """python -m ifg.rsh2dpl.fbf /tmp/rsh2fbf
"""
    # see http://docs.python.org/library/argparse.html
    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument('-t', '--test',
                    action="store_true", default=False, help="run self-tests")
    parser.add_argument('-B', '--buffer-size', metavar='buffer_size', default=0, type=int, help='rsh2fbf circular buffer size' )
    parser.add_argument('-R', '--rolling', default=False, action='store_true', help='rolling records' )
    parser.add_argument('-E', '--earth', default=False, action='store_true', help='yield earth-view frames, implies --rolling' )
    parser.add_argument('-v', '--verbosity', action="count", default=0,
                    help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('src', help = "source file to process")
    # parser.add_argument('-o', '--output',
    #                 help='location to store output')
    # parser.add_argument('-I', '--include',
    #                 action="append", help="include path to append to GCCXML call")
    args = parser.parse_args()



    if args.test:
        # FIXME - run any self-tests
        # import doctest
        # doctest.testmod()
        sys.exit(2)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level = min(3,levels[args.verbosity]))
    LOG.debug('verbosity level is %d' % args.verbosity)

    if not args:
        parser.print_help()
        return
    test_case = """
export STA_BCAST=10.0.1.255:5145
mkdir /tmp/rsh2fbf; cd /tmp/rsh2fbf; RSH2FBF_CIRCULAR_BUFFER_SIZE=256 RSH2FBF_EOB_SLEEP=400 rsh2fbf ~/Data/SHIS/sh070714/dvrvcxv.rrsh ; sleep 15; rm -fr /tmp/rsh2fbf/*
python -m ifg.rsh2dpl.fbf /tmp/rsh2fbf -B 256
"""

    # grab all the low-dimension values
    fnf = lambda fn: fn.endswith('.real4') # fn.startswith('HBB') or fn.startswith('ABB')
    if args.earth:
        args.rolling = True

    f2d = fbf2dpl(args.src, frame_width=32, buffer_size=args.buffer_size, filename_filter = fnf, rolling=args.rolling, rec_gen=True)()

    if args.earth:
        from .scanline import earth_scans
        rolling_frames = f2d
        f2d = earth_scans(rolling_frames)

    for F in f2d:
        print '\n=== record %d~%d' % (F.first_record, F.last_record)
        print 'ABBapexTemp'
        print F.ABBapexTemp
        print 'sceneMirrorAngle'
        print F.sceneMirrorAngle
        sys.stdout.flush()


    return 0


if __name__=='__main__':
    sys.exit(main())
