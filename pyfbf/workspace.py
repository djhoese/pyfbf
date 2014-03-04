"""
Flat binary workspace object




"""

import glob, unittest
from .memfbf import *


class Workspace(object):
    _vars = None
    _dir = None

    def __init__(self, dir='.'):
        self._dir = dir
        self._vars = dict(self._scan_vars())

    def var(self, name):
        v = self._vars.get(name, None)
        if v is not None:
            return v
        g = glob.glob(os.path.join(self._dir, (name + '.*') if '.' not in name else name))
        if len(g) == 1:
            fbf = FBF(g[0])
            fbf.open()
            self._vars[fbf.stemname] = fbf
            return fbf
        raise AttributeError("{0:s} not in workspace".format(name))

    def _scan_vars(self):
        for path in os.listdir(self._dir):
            try:
                x = FBF(os.path.join(self._dir, path))
                yield x.stemname, x
            except ValueError:
                pass

    def variables(self):
        return self._vars

    def __getitem__(self, name):
        return self.var(name)

    def __getattr__(self, name):
        return self.var(name)

    def absorb(self, stemname, nparray):
        """
            Absorb a numpy array into the workspace, resulting in a read-write FBF object.
            Raises an EnvironmentError if there is a collision.
        """
        raise NotImplementedError('Not Yet Implemented')


TEST_WORKSPACE = '/tmp/test_workspace'


class TestWorkspace(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(os.path.exists(TEST_WORKSPACE))
        from pprint import pprint

        ws = Workspace(TEST_WORKSPACE)
        pprint(ws.variables())


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        q = Workspace(argv[1])
        from pprint import pprint

        pprint(dict((x, str(y).split('\n')) for (x, y) in q.variables().items()))
    else:
        unittest.main()
