'''
$Id: numfbf.py,v 1.7 2007/07/31 20:44:52 maciek Exp $
Flat Binary Format python utilities, recoded for speed using numpy.
Maciek Smuga-Otto <maciek@ssec.wisc.edu> 
based on the FBF.py library by Ray Garcia <rayg@ssec.wisc.edu>
'''

import logging
import sys
import os
import numpy
import string
import re

fbf_encodings = dict(
    char1 = 'c', char = 'c',
    int1 = 'b',
    int2 = 'h',
    int4 = 'l', integer4 = 'l',
    real4 = 'f',
    real8 = 'd',
    complex8 = 'F',
    complex16 = 'D'
)

fbf_endian = { 
    '>': string.upper,
    '<': string.lower,
    'big': string.upper,
    'little': string.lower
}

sfx_remap = dict(STA='int1')

# Try to figure out the system's native byte order, defaulting to little-endian if unknown
try:
    byteorder = sys.byteorder
except AttributeError:
    testbytes = struct.pack( '=l', 0xabcd )
    if struct.pack( '<l', 0xabcd ) == testbytes:   byteorder = 'little'
    elif struct.pack( '>l', 0xabcd ) == testbytes: byteorder = 'big'
    else: 
        logging.warning("Unable to identify byte order, defaulting to 'little'.")
        byteorder = 'little'

fbf_endian['native'] = fbf_endian['='] = fbf_endian[ byteorder ]

# reporting format for FBF object
FBF_FMT = """< FBF object:
pathname: %(pathname)s
stemname: %(stemname)s
filename: %(filename)s
dirname : %(dirname)s
type    : %(type)s
grouping: %(grouping)s
byteorder: %(byteorder)s
array_type: %(array_type)s
element_size: %(element_size)s
record_size: %(record_size)s >""" 

class FbfWarning(UserWarning): pass

def array_product(a):
    logging.debug("input to array_product: %s" % a)
    if len(a) == 0: return 1
    elif len(a) == 1: return a[0]
    else:
        return reduce( lambda x,y: x*y, a ) # product of all dimensions
    
class FBF(object):
    def __init__( self, stemname=None, typename=None, grouping=None, dirname='.', byteorder='native' ): 
        if not stemname:
            raise FbfWarning('FBF object must be instantiated with at least a filename')
        if not typename: 
            filename = stemname # filename given as argument: Attach to existing FBF file on disk
            self.attach( filename )
        else:
            # generate a new FBF object, file will need to be explicitly created.
            if not grouping:
                self.grouping = [1]
            self.build( stemname, typename, grouping, dirname, byteorder )
    
    def attach( self, pathname ):
        '''Attach object to existing file on disk'''
        self.pathname = pathname
        self.dirname, self.filename = os.path.split( pathname )
        if not self.dirname: self.dirname = '.'
        
        parts = self.filename.split('.')
        
        self.stemname = parts[0]
        fbftype = parts[1]
        self.grouping = [ int(x) for x in parts[2:] ]
        if not self.grouping:
            self.grouping = [1]
        
        if fbftype.isupper(): 
            self.byteorder='big'
            self.endian='>'
        else: 
            self.byteorder='little'
            self.endian='<'
        
        self.flip_bytes = (self.byteorder != byteorder)
        self.element_size = int(re.findall('\d+',fbftype)[0])
        self.record_elements = array_product(self.grouping)
        self.record_size = self.element_size * self.record_elements        
        self.array_type = fbf_encodings[fbftype]
        self.type = fbftype.lower()
        self.pending_flush = False
        
        return self
    
    def build(self, stemname, typename, grouping=None, dirname='.', byteorder='native' ):
        '''build an FBF descriptor object from scratch.'''
        
        filename = '%s.%s' % ( stemname, fbf_endian[byteorder]( typename ) )
        if grouping and grouping != [1]: 
            filename += '.' + '.'.join( str(x) for x in grouping )
        
        self.attach( os.path.join( dirname, filename ) ) 
    
    def __str__(self):
        return FBF_FMT % vars(self)
    
    def fp(self, mode='rb'):
        try:
            return self.file
        except AttributeError:
            return open(self.pathname, mode)
    
    def open( self, mode='rb' ): 
        self.file = self.fp(mode)
        return self.file
    
    def create( self, mode='w+b' ):
        self.file = self.fp(mode)
        return self.file
    
    def close( self ):
        if self.pending_flush:
            try: self.file.flush()
            except: pass
        try: 
            del self.file
            return 0
        except AttributeError:
            return 1
    
    def length( self ):
        if self.pending_flush:
            try: self.file.flush()
            except: pass
        
        self.pending_flush = False
        return os.stat(self.pathname).st_size / self.record_size
    
    def __len__( self ):
        return self.length()
    
    def __getitem__( self, idx ):
        '''obtain records from an FBF file - pass in either an FBF object or an FBF file name.
        Index can be a regular python slice or a 0-based index'''
        if not hasattr(self, 'data'):
            fp = self.fp()
            fp.seek(0)
            self.data = numpy.fromfile( fp, self.array_type ).reshape( [self.length()] + self.grouping[::-1] )
            if self.flip_bytes: 
                self.data.dtype = self.data.dtype.newbyteorder()
                # hack: register byte order as swapped in the numpy array without flipping any actual bytes

        logging.debug("len: %s grouping: %s" % (self.length(),self.grouping))
        logging.debug("data shape: %s returned: %s" % (self.data.shape,self.data[idx].shape))
        return self.data[idx]
    

def build( stemname, typename, grouping=None, dirname='.', byteorder='native' ): 
    return FBF( stemname, typename, grouping, dirname, byteorder )

def read( fbf, start=1, end=0 ):
    '''legacy API call uses 1-based indexing, so read(1,-1) reads in the whole dataset'''
    if isinstance( fbf, str ): fbf = FBF(fbf)
    idx = start-1
    if end < -1:
        idx = slice(start-1, end+1)
    elif end == -1:
        idx = slice(start-1,None) # special case: slice(n,0) is not equivalent
    elif end > 0:
        idx = slice(start-1, end-1)
    
    return fbf[idx]

def write( fbf, idx, data ):
    '''write records to an FBF file - pass in either an FBF object or an FBF file name
    The index is a 1-based int (as in Fortran and Matlab). 
    Data must be an appropriately shaped and typed numpy array'''
    # FIXME: can only write in native byte order as far as I can tell.
    if isinstance( fbf, str ): fbf = FBF(fbf)
    if ( array_product(data.shape) % fbf.record_elements ) != 0:
        raise FbfWarning("data incorrectly shaped for write")
        
    index = min( idx-1, fbf.length() )
    fbf.file.seek( index * fbf.record_size )
    data.tofile(fbf.file)
    fbf.pending_flush = True
    return 1

def block_write( fbf, idx, data ):
    return write( fbf, idx, data )
    
def info(name):
    return FBF(name)

FBF.read = read
FBF.write = write
FBF.block_write = block_write

def test():
    import doctest
    doctest.testfile( "numfbf.doctest" )
    
if __name__ == '__main__':
    test()
