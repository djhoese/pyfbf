#!/usr/bin/env python


import os
import sys
import logging
from subprocess import Popen as popen
from Queue import Queue, Empty as EmptyException
from glob import glob
from numpy import append

from .slicer import fbf_slicer

LOG = logging.getLogger(__name__)


class fbf2dpl(object):
    """convert a FBF non-circular working directory into a DPL frame stream
    """
    _slicer = None            # fbf_slicer instance 
    _eob_sleep = None         # minimum time delay between frames in seconds
    
    def __init__(self, work_dir, eob_sleep=None, filename_filter=None):
        assert(os.path.isdir(work_dir))
        self._eob_sleep = eob_sleep
        self._slicer = fbf_slicer(work_dir, 0, filename_filter)
        
    def close(self):
        return

    def __call__(self):
        rec = 1
        next_time = time.time() if self._eob_sleep else None
        while True:
            if self._eob_sleep:
                now = time.time()
                delta = next_time - now
                if delta>0.0:
                    time.sleep(delta)
                next_time = now + self._eob_sleep
            data = self._slicer(rec)
            if data is None:
                return
            yield data
            rec += 1
