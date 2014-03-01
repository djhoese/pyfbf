'''
$Id$
Flat Binary Format python utilities, recoded for speed using numpy.
Maciek Smuga-Otto <maciek@ssec.wisc.edu> 
based on the FBF.py library by Ray Garcia <rayg@ssec.wisc.edu>
'''

import logging
import sys
import os
import numpy
import mmap
import re
from functools import reduce

LOG = logging.getLogger(__name__)

fbf_encodings = dict(
    char1 = 'c', char = 'c',
    int1 = 'b', uint1 = 'B',
    int2 = 'h', uint2 = 'H',
    int4 = 'i', integer4 = 'i', uint4 = 'I',
    int8 = 'q', uint8 = 'Q',
    real4 = 'f',
    real8 = 'd',
    complex8 = 'F',
    complex16 = 'D'
)

FBF_ENDIAN = {
    '>': str.upper,
    '<': str.lower,
    'big': str.upper,
    'little': str.lower
}


# FIXME: resolve fbf_encodings vs SUFFIX_TO_DTYPE usage

FBF_FLOAT32   = "real4"
FBF_FLOAT64   = "real8"
FBF_INT8      = "int1"
FBF_INT16     = "int2"
FBF_INT32     = "int4"
FBF_INT64     = "int8"
FBF_UINT8     = "uint1"
FBF_UINT16    = "uint2"
FBF_UINT32    = "uint4"
FBF_UINT64    = "uint8"

SUFFIX_TO_DTYPE = {
        FBF_FLOAT32   : numpy.float32,
        FBF_FLOAT64   : numpy.float64,
        FBF_INT8      : numpy.int8,
        FBF_INT16     : numpy.int16,
        FBF_INT32     : numpy.int32,
        FBF_INT64     : numpy.int64,
        FBF_UINT8     : numpy.uint8,
        FBF_UINT16    : numpy.uint16,
        FBF_UINT32    : numpy.uint32,
        FBF_UINT64    : numpy.uint64
        }

# FIXME avoid this __name__ nonsense
SDTYPE_TO_SUFFIX = dict((v.__name__,k) for (k,v) in SUFFIX_TO_DTYPE.items())   # FUTURE: really would prefer not having a string key


sfx_remap = dict(STA='int1')

RE_FBF = re.compile(r'^.*?\.[A-Za-z]\w+(?:\.\d+)*$')

# Try to figure out the system's native byte order, defaulting to little-endian if unknown
try:
    BYTEORDER = sys.byteorder
except AttributeError:
    import struct
    testbytes = struct.pack( '=l', 0xabcd )
    if struct.pack( '<l', 0xabcd ) == testbytes:   BYTEORDER = 'little'
    elif struct.pack( '>l', 0xabcd ) == testbytes: BYTEORDER = 'big'
    else: 
        LOG.warning("Unable to identify byte order, defaulting to 'little'.")
        BYTEORDER = 'little'

FBF_ENDIAN['native'] = FBF_ENDIAN['='] = FBF_ENDIAN[ BYTEORDER ]

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

class FbfWarning(UserWarning):
    pass


def filename(stem, dtype, shape=None):
    """build a filename, given a stem, element datatype, and record array shape
    filename('mydata', data.dtype, data.shape)
    """
    typename = SDTYPE_TO_SUFFIX[str(dtype)]
    fn = '%s.%s' % ( stem, FBF_ENDIAN[BYTEORDER](typename) )
    if shape is not None:
        shape = tuple(shape)
    if shape is None or shape == (1,):
        return fn
    return fn + '.' + '.'.join(str(x) for x in reversed(shape))


def array_product(a):
    LOG.debug("input to array_product: %s" % repr(a))
    if len(a) == 0: return 1
    elif len(a) == 1: return a[0]
    else:
        return reduce( lambda x,y: x*y, a ) # product of all dimensions


class FBF(object):
    # filename attributes
    pathname = None  # full path (redundant?)
    dirname = None
    filename = None
    stemname = None  # filename before suffix (redundant?)

    # byte order attributes
    grouping = None
    byteorder = None  # 'big', 'little'  (redundant)
    endian = None  # struct '>' or '<'
    flip_bytes = False  # bool (redundant?)

    # array attributes
    element_size = 0  # int
    record_elements = 0  # int
    record_size = None  # int (redundant)
    array_type = None  # struct code (redundant)
    type = None  # str: real4, int8, etc
    dtype = None  # numpy dtype (redundant)

    # file attributes
    pending_flush = False  # bool, whether or not a .flush is needed prior to checking file length
    file = None # file object
    mode = None # open() mode, e.g. 'rb'
    mmap = None # mmap object
    mmap_length = 0 # number of records available in mmap (redundant?)
    data = None # numpy object representing mmap


    def __init__( self, stem_or_filename=None, typename=None, grouping=None, dirname='.', byteorder='native' ):
        if not stem_or_filename:
            raise FbfWarning('FBF object must be instantiated with at least a filename')
        elif not typename:
            self.attach( stem_or_filename )
        else:
            # generate a new FBF object, file will need to be explicitly created.
            if not grouping:
                self.grouping = [1]
            self.build( stem_or_filename, typename, grouping, dirname, byteorder )
    
    def attach( self, pathname ):
        '''Attach object to existing file on disk'''
        if not RE_FBF.match(pathname):
            raise ValueError('%s is not a valid FBF pathname' % pathname)
        self.pathname = pathname
        self.dirname, self.filename = os.path.split( pathname )
        if not self.dirname: self.dirname = '.'
        
        parts = self.filename.split('.')
        
        self.stemname = parts[0]
        fbftype = sfx_remap.get(parts[1].upper(), parts[1])
        self.grouping = [ int(x) for x in parts[2:] ]
        if not self.grouping:
            self.grouping = [1]
        
        if fbftype.isupper(): 
            self.byteorder='big'
            self.endian='>'
        else: 
            self.byteorder='little'
            self.endian='<'
        
        self.flip_bytes = (self.byteorder != BYTEORDER)
        try:
            self.element_size = int(re.findall('\d+',fbftype)[0])
        except IndexError:
            raise ValueError('cannot attach %s as valid FBF' % pathname)
        self.record_elements = array_product(self.grouping)
        self.record_size = self.element_size * self.record_elements        
        self.array_type = fbf_encodings[fbftype]
        self.type = fbftype.lower()
        self.dtype = SUFFIX_TO_DTYPE[self.type]
        self.pending_flush = False
        
        return self
    
    def build(self, stemname, typename, grouping=None, dirname='.', byteorder='native' ):
        '''build an FBF descriptor object from scratch.'''
        
        filename = '%s.%s' % ( stemname, FBF_ENDIAN[byteorder]( typename ) )
        if grouping and grouping != [1]: 
            filename += '.' + '.'.join( str(x) for x in grouping )
        
        self.attach( os.path.join( dirname, filename ) ) 
    
    def __str__(self):
        return FBF_FMT % vars(self)
    
    def fp(self, mode='rb'):
        fob = getattr(self, 'file', None)
        if not fob: 
            fob = self.file = open(self.pathname, mode)
            self.mode = mode
        return fob
    
    def open( self, mode='rb' ): 
        self.file = self.fp(mode)
        return self.file
    
    def create( self, mode='w+b' ):
        self.file = self.fp(mode)
        return self.file

    def _flush_if_needed(self):
        if self.pending_flush and self.file:
            self.file.flush()
            self.pending_flush = False

    def close( self ):
        self._flush_if_needed()
        if not self.file:
            return 1
        self.file = None
        return 0

    def length( self ):
        self._flush_if_needed()
        return int(os.stat(self.pathname).st_size / self.record_size)
    
    def __len__( self ):
        return self.length()
    
    def __getitem__( self, idx ):
        '''obtain records from an FBF file - pass in either an FBF object or an FBF file name.
        Index can be a regular python slice or a 0-based index'''
        length = self.length()
        if not hasattr(self, 'data') or getattr(self,'mmap_length',0)!=length:
            fp = self.fp()
            fp.seek(0)
            shape = [length] + self.grouping[::-1]
            LOG.debug('mapping with shape %r' % shape)
            if '+' in self.mode or 'w' in self.mode:
                access = mmap.ACCESS_WRITE
            else:
                access = mmap.ACCESS_READ
            self.mmap = mmap.mmap(fp.fileno(), 0, access=access)
            self.data = numpy.ndarray( buffer = self.mmap, shape=shape, dtype=self.endian + self.array_type )
            self.mmap_length = length
#             if self.flip_bytes: 
#                 self.data.dtype = self.data.dtype.newbyteorder()
                # hack: register byte order as swapped in the numpy array without flipping any actual bytes

        #LOG.debug("len: %s grouping: %s" % (self.length(),self.grouping))
        #LOG.debug("data shape: %s returned: %s" % (self.data.shape,self.data[idx].shape))
        return self.data[idx]
    

def build( stemname, typename, grouping=None, dirname='.', byteorder='native' ):
    """
    DEPRECATED: procedural API for creating FBF objects
    """
    return FBF( stemname, typename, grouping, dirname, byteorder )

def _one_based_slice(fbf, start, end):
    if start == end == 0:
        start, end = 1, -1
    idx = start-1
    if end < -1:
        idx = slice(start-1, end+1)
    elif end == -1:
        idx = slice(start-1,None) # special case: slice(n,0) is not equivalent
    elif end > 0:
        idx = slice(start-1, end-1)
    return idx

def read( fbf, start=0, end=0 ):
    '''DEPRECATED: legacy API call uses 1-based indexing, so read(1,-1) reads in the whole dataset'''
    if isinstance( fbf, str ):
        fbf = FBF(fbf)
    idx = _one_based_slice(fbf, start, end)

    return fbf[idx]


def extract_indices_to_file( inp, indices, out_file ):
    "DEPRECATED: 0-based index list is transcribed to a new output file in list order"
    if isinstance(inp, str):
        inp = FBF(inp)
    if isinstance(out_file, str):
        fout = open( out_file, 'wb' )
        shouldclose=True
    else:
        shouldclose=False
    for r in indices:
        inp[r].tofile(fout)
    if shouldclose: fout.close()

def write( fbf, idx, data ):
    '''DEPRECATED: write records to an FBF file - pass in either an FBF object or an FBF file name
    The index is a 1-based int (as in Fortran and Matlab). 
    Data must be an appropriately shaped and typed numpy array'''
    # FIXME: can only write in native byte order as far as I can tell.
    # FIXME: needs to convert to expected format, array order and contiguity, or raise an error on bad content
    if isinstance( fbf, str ): fbf = FBF(fbf)
    if ( array_product(data.shape) % fbf.record_elements ) != 0:
        # FIXME: this should really check the shape and not just that record_elements divides length
        raise FbfWarning("data incorrectly shaped for write")

    index = min( idx-1, fbf.length() )
    if index != idx-1:
        raise FbfWarning('data being written to record %d instead of %d' % (index+1, idx))
    fbf.file.seek( index * fbf.record_size )
    data.tofile(fbf.file)
    fbf.pending_flush = True
    return 1

def block_write( fbf, idx, data ):
    """
    DEPRECATED: block write to a file using one-based indexing
    """
    return write( fbf, idx, data )
    
def info(name):
    return FBF(name)

FBF.read = read
FBF.write = write
FBF.block_write = block_write

def test():
    import doctest
    logging.basicConfig(level = logging.DEBUG)
    doctest.testfile( "numfbf.doctest" )
    
if __name__ == '__main__':
    test()
