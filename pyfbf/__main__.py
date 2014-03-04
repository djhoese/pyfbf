import sys, unittest
from pprint import pprint
from .workspace import Workspace

if __name__=='__main__':
    for dn in (sys.argv[1:]):
        w = Workspace(dn)
        pprint(w.variables())

