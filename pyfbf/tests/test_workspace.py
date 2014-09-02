#!/usr/bin/env python
"""Tests for the workspace.py module.
"""
import os
import shutil
import unittest
import numpy
from pyfbf.workspace import Workspace

TEST_WORKSPACE = '/tmp/test_workspace'


class TestWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.remove_dir = False
        if not os.path.exists(TEST_WORKSPACE):
            cls.remove_dir = True
            os.makedirs(TEST_WORKSPACE)
            numpy.arange(10).astype(numpy.float32).tofile(os.path.join(TEST_WORKSPACE, "test1.real4"))

    @classmethod
    def tearDownClass(cls):
        if cls.remove_dir:
            shutil.rmtree(TEST_WORKSPACE)

    def test_simple(self):
        self.assertTrue(os.path.exists(TEST_WORKSPACE))
        from pprint import pprint

        ws = Workspace(TEST_WORKSPACE)
        pprint(ws.variables())
