#!/usr/bin/env python
"""Tests for the memfbf.py module.
"""
import os
import unittest
import logging
import numpy as np
from pyfbf.memfbf import FBF, read

LOG = logging.getLogger(__name__)


class TestFBF(unittest.TestCase):
    def setUp(self):
        # FUTURE: consolidate test patterns where we can build them into here, then knock them down later
        pass

    def test_create(self):
        LOG.info('create')
        fbf = FBF('foo', 'real4', [15], writable=True)
        fbf.create(records=3)
        LOG.info('shape {0}'.format(repr(fbf.shape)))
        self.assertEqual(fbf.path, './foo.real4.15')
        self.assertEqual(len(fbf), 3)
        self.assertEqual(fbf.ndim, 2)
        a = np.array([np.float32(x) / 2 for x in range(45)], 'f')
        a = a.reshape([3, 15])
        LOG.info(fbf[0])
        fbf[0:3] = a[0:3]
        LOG.info(fbf[:])
        fbf[3:6] = a[0:3]
        os.unlink(fbf.path)

    def test_create_unicode(self):
        LOG.info('create_unicode')
        fbf = FBF(u'foo', u'real4', [15], writable=True)
        fbf.create(records=3)
        LOG.info('shape {0}'.format(repr(fbf.shape)))
        self.assertEqual(fbf.path, u'./foo.real4.15')
        self.assertEqual(len(fbf), 3)
        self.assertEqual(fbf.ndim, 2)
        a = np.array([np.float32(x) / 2 for x in range(45)], 'f')
        a = a.reshape([3, 15])
        LOG.info(fbf[0])
        fbf[0:3] = a[0:3]
        LOG.info(fbf[:])
        fbf[3:6] = a[0:3]
        os.unlink(fbf.path)

    def test_close_open(self):
        LOG.info('classic')
        foo = FBF('foo', 'real4', [15], writable=True)
        foo.create(records=2)
        self.assertEqual(2, len(foo))
        a = np.array([float(x) / 2 for x in range(45)], 'f').reshape((3, 15))
        foo[0:3] = a
        self.assertEqual(3, len(foo))
        self.assertEqual(foo.stemname, 'foo')
        del foo
        foo = FBF('foo.real4.15')
        self.assertTrue((foo[0:3] == a).all())
        with self.assertRaises(IndexError):
            q = foo[4]
        self.assertTrue((read(foo.path, 1, -1) == a).all())
        os.unlink(foo.path)

    def test_malformed_filenames(self):
        LOG.info('malformed - check filename parsing')
        with self.assertRaises(ValueError):
            foo = FBF('any.txt')
        with self.assertRaises(ValueError):
            foo = FBF('/path/to/my/dog')

    def test_append_basics(self):
        LOG.info('append - test basics of append-mode FBF')
        foo = FBF('foo', 'real4', [15], writable=True)
        foo.create(records=2)
        self.assertEqual(2, len(foo))
        a = np.array([float(x) / 2 for x in range(45)], np.float64).reshape((3, 15))
        foo[0:3] = a
        self.assertEqual(3, len(foo))
        self.assertEqual(foo.stemname, 'foo')
        foo.append(a)
        self.assertTrue((foo[3:6] == a).all())
        os.unlink(foo.path)

    def test_append_just_real4s(self):
        LOG.info('append2 - test append-only FBF and simple .real4 handling')
        foo = FBF('foo', 'real4', writable=True)
        a = np.array([float(x) / 2 for x in range(45)], np.float64)
        foo.append(a)
        nrecs = foo.append(a)
        self.assertEqual(nrecs, len(a)*2)
        ar = np.concatenate([a, a])
        # LOG.debug(ar[:])
        LOG.debug(repr(foo.dtype))
        self.assertTrue(foo.data is None)
        foo.open(mode='r')
        LOG.info(foo[:])
        self.assertEqual(len(foo), len(a)*2)
        boo = np.all(ar == foo[:])
        LOG.debug(repr(boo))
        self.assertTrue(boo)
        os.unlink(foo.path)

    def test_index_array_access(self):
        foo = FBF('foo', 'real8', [15], writable=True)
        foo.create(records=6)
        a = np.array([float(x) / 2 for x in range(45)], np.float64).reshape((3, 15))
        foo[0:3] = a
        foo[3:6] = a
        dexy = [0, 3, 4]
        truth = a[[0, 0, 1]]
        foodexy = foo[dexy]
        self.assertTrue((foodexy[:] == truth).all())
        booly = [False, True, True, False, True, False]
        foolery = foo[booly]
        self.assertEqual(foolery.shape, (3, 15))
        foodexy = foo[[1, 2, 4]]
        self.assertTrue((foolery == foodexy).all())
        foodexy = foo[::2]
        foolery = a[[0, 2, 1]]
        self.assertTrue((foolery == foodexy).all())
        foodexy = foo[1::2]
        foolery = a[[1, 0, 2]]
        self.assertTrue((foolery == foodexy).all())
        os.unlink(foo.path)


def test(debug=False):
    logging.basicConfig(level=logging.DEBUG if debug else logging.ERROR)
    unittest.main()

if __name__ == '__main__':
    test(debug='DEBUG' in os.environ)
