#!/usr/bin/env python

from . import memfbf
from numpy import append

import os
import logging
from glob import glob

LOG = logging.getLogger(__name__)


class SlicerFrame(dict):
    pass


class FBFSlicer(object):
    """Given a workspace directory of flat binary files, grab all useful filenames and return a record of data at a
    time as a python dictionary.
    """
    def __init__(self, work_dir, buffer_size=0, filename_filter=None):
        """Initialize slicer object parameters.

        :param work_dir: Workspace directory of flat binary files to read
        :param buffer_size: Circular buffer size or 0 for non-circular buffers/FBFs
        :param filename_filter: Filter function that returns True if the provided file should be opened for reading.
                                Should return False otherwise.
        """
        self._wd = work_dir
        self._buffer_size = buffer_size
        self._open_files = dict()
        if filename_filter is None:
            filename_filter = lambda filename: True
        self.should_include = filename_filter

    def _update_open_files(self):
        for fn in glob(os.path.join(self._wd, '*')):
            if fn not in self._open_files and self.should_include(os.path.split(fn)[-1]):
                LOG.debug('opening %s' % fn)
                try:
                    nfo = memfbf.FBF(fn)
                except Exception as oops:
                    nfo = None
                    LOG.info('%s could not be opened as FBF' % fn)
                    LOG.debug(repr(oops))
                LOG.debug('found new file %s' % fn)
                self._open_files[fn] = nfo

    def __call__(self, first_record, last_record=None):
        """Retrieve a slice of a FBF directory using inclusive 1-based record number range, noting
        that last-first+1 records are returned.
        """
        last_record = first_record if last_record is None else last_record
        if not self._open_files:
            self._update_open_files()
        data = SlicerFrame()
        for name, nfo in self._open_files.items():
            if nfo is not None:
                # note we use % in order to deal with
                # wavenumber files that are only ever 1 record long
                # circular buffers which are fixed length files
                file_len = nfo.length()
                # check for non-circular buffer case and going off the end of the file
                # note use of > since record numbers are 1-based
                if (self._buffer_size == 0) and (file_len != 1) and (first_record > file_len or last_record > file_len):
                    LOG.warning('%s: length is %d but start-end is %d-%d' % (name, file_len, first_record, last_record))
                    return None
                # check for circular buffers that aren't preallocated properly
                if self._buffer_size > 0 and file_len not in (1, self._buffer_size):
                    LOG.info('buffer file %s size mismatch (%d != %d)! ignoring' % (name, file_len, self._buffer_size))
                else:
                    # 0-based circular buffer
                    first_index = (first_record - 1) % file_len
                    last_index = (last_record - 1) % file_len
                    if last_index >= first_index:
                        # Records are in one continuous line
                        idx = slice(first_index, last_index + 1)  # +1 to include last item
                        data[nfo.stemname] = nfo[idx]
                    else:
                        # Records are on two ends of the circular buffer
                        idx1 = slice(first_index, self._buffer_size)
                        idx2 = slice(0, last_index + 1)  # +1 to include last item
                        arr1 = nfo[idx1]
                        arr2 = nfo[idx2]
                        data[nfo.stemname] = append(arr1, arr2, axis=0)

        return data

