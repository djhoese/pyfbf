"""
Flat binary workspace object




"""

import os,glob

from .memfbf import *

# FUTURE: workspace should return numpy.memmap arrays

class Workspace(object):
    def __init__(self, dir='.'):
        self._dir=dir

    def var(self, name):
        g = glob.glob( os.path.join(self._dir, (name + '.*') if '.' not in name else name) )
        if len(g)==1:
            fp = FBF(g[0])
            fp.open()
            setattr(self,fp.stemname,fp)
            return fp
        raise AttributeError("%s not in workspace" % name)
    
    def vars(self):

        for path in os.listdir(self._dir):
            try:
                x = info(path)
                yield x.stemname, x
            except ValueError:
                pass
            
    def variables(self):
        return dict(self.vars())

    def __getitem__(self,name):
        return self.var(name)

    def __getattr__(self, name):
        return self.var(name)

    def absorb(self, stemname, nparray):
        """
            Absorb a numpy array into the workspace, resulting in a read-write FBF object.
            Raises an EnvironmentError if there is a collision.
        """
        raise NotImplementedError('Not Yet Implemented')
        

    
if __name__=='__main__':
    from sys import argv
    where = '.' if len(argv)==1 else argv[1]
    q = Workspace(where)
    from pprint import pprint
    pprint(dict((x,str(y).split('\n')) for (x,y) in q.vars()))
