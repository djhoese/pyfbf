#!/usr/bin/env python

import os
import sys
import logging
from subprocess import Popen as popen
from Queue import Queue, Empty as EmptyException
from glob import glob
from numpy import append

import keoni.fbf as fbf

LOG = logging.getLogger(__name__)


class rshdata(object):  
    "read all the values from current workspace directory into attributes"    
    pass


class fbf_slicer(object):
    """given a workspace directory of FBF files, grab all useful filenames and return a record of data at a time
    """
    _wd = None
    _buffer_size = 0   # number of records to expect in circular buffer files
    _open_files = None
    should_include = None # callable, returns True if a given filename should be included in the frame 
    
    def __init__(self, work_dir, buffer_size, filename_filter=None):
        self._wd = work_dir
        self._buffer_size = buffer_size
        self._open_files = dict()
        if filename_filter is None:
            filename_filter = lambda filename: True
        self.should_include = filename_filter
    
    def _update_open_files(self):
        for fn in glob( os.path.join(self._wd, '*') ):
            if fn not in self._open_files and self.should_include(os.path.split(fn)[-1]):
                LOG.debug('opening %s' % fn)
                try:
                    nfo = fbf.FBF(fn)
                except Exception as oops:
                    nfo = None    
                    LOG.info('%s could not be opened as FBF' % fn)
                    LOG.debug(repr(oops))
                LOG.debug('found new file %s' % fn)
                self._open_files[fn] = nfo

    def __call__(self, start_record, end_record=None):
        "retrieve a slice of a FBF directory using 1-based record range"
        end_record = start_record + 1 if end_record is None else end_record
        if not self._open_files:
            self._update_open_files()
        data = rshdata()
        for name, nfo in self._open_files.items():
            if nfo is not None: 
                # note we use % in order to deal with
                # wavenumber files that are only ever 1 record long
                # circular buffers which are fixed length files 
                len = nfo.length()
                # check for non-circular buffer case and going off the end of the file
                # note use of > since record numbers are 1-based
                if self._buffer_size ==0 and (start_record > len or end_record > len):
                    return None
                # check for circular buffers that aren't preallocated properly
                if self._buffer_size > 0 and len not in (1, self._buffer_size):                    
                    LOG.info('buffer file %s size mismatch (%d != %d)! ignoring' % (name,len,self._buffer_size))
                else:
                    # 0-based circular buffer
                    start_index = (start_record - 1) % len
                    end_index = (end_record - 1) % len
                    if end_index >= start_index:
                        # Records are in one continuous line
                        idx = slice(start_index, end_index + 1) # +1 to include last item
                        setattr(data, nfo.stemname, nfo[idx])
                    else:
                        # Records are on two ends of the circular buffer
                        idx1 = slice(start_index, self._buffer_size)
                        idx2 = slice(0, end_index + 1) # +1 to include last item
                        arr1 = nfo[idx1]
                        arr2 = nfo[idx2]
                        setattr(data, nfo.stemname, append(arr1, arr2, axis=0))

        return data
