#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyfbf.concatenate
=================

PURPOSE
concatenate a bunch of FBF workspaces into one directory, safely


REQUIRES
numpy

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import argparse
import logging
import os
import sys
import unittest
from functools import reduce

import numpy as np

from .workspace import Workspace

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)


def _minrecs(W, tail_variables):
    try:
        nrecs = [len(W[varname]) for varname in tail_variables]
        return min(nrecs)
    except KeyError as not_available:
        return 0


def _relevant_variables(V, minrecs):
    for k, v in V.items():
        if len(v) >= minrecs:
            yield k


def concatenate(paths, tail_variables, output='.', dry_run=False):
    """
    build master variable list of variables and their record counts for each workspace in paths
    for each workspace in paths
      take minimum number of records among the tail variables => minrecs
        ignoring (with warning) workspaces that do not have all of the tail variables
    eliminate from master variable list any variable not having minrecs for its workspace
    for each variable in master list
      for each workspace
        concatenate data to a variable in the cwd, truncating to minrecs for that workspace
    :param paths: workspace paths to concatenate, in order
    :param final_stems: variable names which determine the minimum number of records
    :param output: optional output directory to create (if needed) and write to
    :param dry_run: run without creating output files
    :return: total number of records
    """
    Ws = [Workspace(p) for p in paths]  # workspaces
    Vs = [w.variables() for w in Ws]  # variables for workspaces
    Ns = [_minrecs(w, tail_variables) for w in Ws]  # minimum records per workspace, per tail variables
    relevant_variable_names = [set(_relevant_variables(v, n)) for v, n in zip(Vs, Ns)]  # variable names to consider
    common_variable_names = reduce(lambda a, b: a & b, relevant_variable_names)
    LOG.info('variables of interest: ' + ', '.join(sorted(common_variable_names)))
    LOG.info('transfer from workspaces: ' + ', '.join('%s: %d' % (w.path, n) for (w,n) in zip(Ws,Ns)))
    if not os.path.isdir(output) and not dry_run:
        os.makedirs(output)
    for var_name in sorted(common_variable_names):
        fob = None
        LOG.debug(var_name)
        for V,n in zip(Vs,Ns):
            if n <= 0:
                continue
            v = V[var_name]
            if fob is None:
                filename = os.path.split(v.path)[-1]
                if not dry_run:
                    fob = open(os.path.join(output, filename), 'ab')
            data = v[:n]
            if not dry_run:
                data.tofile(fob)
            v.close()
    total = int(np.sum(Ns))
    LOG.info("total resulting records: %d" % total)
    return total


class tests(unittest.TestCase):
    data_file = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

    def setUp(self):
        pass

    def test_rdrs(self):
        from glob import glob
        dirnames = list(glob('rdr????????T??????sdr*'))
        dirnames.sort()
        concatenate(dirnames, ['zfliLW', 'zfliMW', 'zfliSW'], dry_run=True)



def _debug(type, value, tb):
    "enable with sys.excepthook = debug"
    if not sys.stdin.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
    parser = argparse.ArgumentParser(
        description="concatenate multiple FBF workspaces into one, for variables matching length criteria",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    parser.add_argument('-t', '--dry-run', dest='dryrun', action='store_true',
                        help="walk through but do not write data files")
    parser.add_argument('-r', '--records', dest='records', action='append',
                        help="specify variable name to use for identify records in a given workspace")
    parser.add_argument('-o', '--output', dest='output',
                        help="optional output directory to write to (create if needed)")
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('inputs', nargs='*',
                        help="input files to process")
    args = parser.parse_args()

    if args.debug:
        sys.excepthook = _debug

    if not args.inputs:
        logging.basicConfig(level=logging.DEBUG)
        unittest.main()
        return 0

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    n_written = concatenate(args.inputs, args.records, output=args.output, dry_run=args.dryrun)
    print('Wrote %d records.' % n_written)

    for pn in args.inputs:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
