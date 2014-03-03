'''
$Id$
Flat Binary Format python utilities, recoded for speed using numpy.
Ray Garcia <rayg@ssec.wisc.edu>
'''

import os, sys, logging, unittest
import numpy as np
import re
from functools import reduce

LOG = logging.getLogger(__name__)

RE_FILENAME=re.compile(r'^(?P<stem>\w+)\.(?:(?P<lend>[a-z][a-z0-9]+)|(?P<bend>[A-Z][A-Z0-9]+))(?:\.(?P<dims>[\.0-9]+))?$')
# foo.real4.20.10
#  stem: 'foo'
#  end (bend/lend): 'real4'
#  dims: 20.10


# ref http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing
SUFFIX_TO_TYPESTRING = dict(
    char1 = 'c', char = 'c',
    int1 = 'i1', uint1 = 'u1', sta = 'i1',
    int2 = 'i2', uint2 = 'u2',
    int4 = 'i4', uint4 = 'u4',
    int8 = 'i8', uint8 = 'u8',
    real4 = 'f4',
    real8 = 'f8',
    complex8 = 'c8',
    complex16 = 'c16',
)

TYPESTRING_TO_SUFFIX = dict((v,k) for (k,v) in SUFFIX_TO_TYPESTRING.items())


def _dtype_from_regex_groups(stem=None, lend=None, dims=None, bend=None):
    """
    return a numpy dtype object from a suffix string,
    describing the composition and shape of individual records

    :param stem: stem string
    :param lend: little-endian ending, e.g. real4
    :param dims: string of dimensions, e.g. 20.10
    :param bend: big-endian ending, e.g. REAL4
    """
    shape = 1
    if dims:
        shape = tuple(reversed(tuple(map(np.uint32, dims.split('.')))))
    try:
        if bend is not None:
            descr = '>' + SUFFIX_TO_TYPESTRING[bend.lower()]
        elif lend is not None:
            descr = '<' + SUFFIX_TO_TYPESTRING[lend]
        else:
            raise ValueError('no suffix provided??')
    except KeyError:
        raise ValueError('%s%s is not a known suffix' % (bend, lend))
    return np.dtype((descr, shape))


def _suffix_from_dtype(data, shape=None, multiple_records=False):
    """
    return the string file suffix for a file given the data type object representing one record
    :param data: numpy dtype or array
    :param shape: if data is a simple type (numpy.int32 for instance) then this is shape of a record
    :param multiple_records: if data is an array, this determines whether to treat the first dimension as the record dimension
    :returns: string file suffix without '.' on front, e.g. "real4.20.10"
    """
    if isinstance(data, np.ndarray):
        data = np.dtype((data.dtype, data.shape[1:] if multiple_records else data.shape))
    elif not isinstance(data, np.dtype) and (shape is not None):
        data = np.dtype((data, shape))
    elif not isinstance(data, np.dtype):
        data = np.dtype((data, 1))

    subd = data.subdtype
    if not subd:
        descr, = data.base.descr
        shape = []
    else:
        descr, = subd[0].base.descr
        shape = subd[1]
    typestring = descr[1]

    assert(typestring[0] in '<|>')
    suffix = TYPESTRING_TO_SUFFIX[typestring[1:]]
    if typestring[0]=='>' or (typestring[0]=='|' and sys.byteorder=='big'):
        suffix = suffix.upper()
    if shape:
        suffix += '.' + '.'.join(map(str, reversed(shape)))
    return suffix


class TestBasics(unittest.TestCase):
    def setUp(self):
        pass

    def test_suffix(self):
        sfx = _suffix_from_dtype(np.float64, (10,20))
        self.assertEqual(sfx, 'real8.20.10')

    def test_array2suffix(self):
        t = np.dtype(('f8', (20, 10)))
        a = np.zeros((5,), dtype=t)
        sfx = _suffix_from_dtype(a)
        self.assertEqual(sfx, 'real8.10.20.5')
        sfx = _suffix_from_dtype(a, multiple_records=True)
        self.assertEqual(sfx, 'real8.10.20')

    def test_dtype(self):
        m = RE_FILENAME.match('foo.real8.10.20')
        dt = _dtype_from_regex_groups(**m.groupdict())
        a = np.zeros((5,), dt)
        self.assertEqual(a.shape, (5,20,10))
        self.assertEqual(a.dtype, np.float64)








def filename(stem, dtype, shape=None, multiple_records=False):
    """build a filename, given a stem, element datatype, and record array shape
    filename('mydata', data.dtype, data.shape)
    """
    suffix = _suffix_from_dtype(dtype, shape, multiple_records)
    return stem + '.' + suffix


BYTE_ORDER_BIG = ('<', str.lower)
BYTE_ORDER_LITTLE = ('>', str.upper)
BYTE_ORDER_NATIVE = BYTE_ORDER_BIG if sys.byteorder == 'big' else BYTE_ORDER_LITTLE


BYTE_ORDER_TABLE = {'big': BYTE_ORDER_BIG,
                   'little': BYTE_ORDER_LITTLE,
                   '>': BYTE_ORDER_BIG,
                   '<': BYTE_ORDER_LITTLE
}


def _dtype_from_path(pathname):
    """
    pull the numpy dtype out of a pathname
    :param pathname: path to look at
    :return: record dtype
    """
    dn,fn = os.path.split(pathname)
    return _dtype_from_regex_groups(**RE_FILENAME.match(fn).groupdict())


def _construct_from_options(stem_or_filename=None, typename=None, grouping=None, dirname=None, byteorder=None):
    """
    Convert a creation specification into a (pathname, record-dtype) pair
    :param stem_or_filename: /path/to/name.real4, name.real4, or name
    :param typename: FBF type name, e.g. 'real4', in the case that stem_or_filename doesn't include it
    :param grouping: tuple of dimensions in access order, e.g. (768,1024) for a 1024w X 768h image array with rows contiguous
    :param dirname: where to place the file if it's being created and not already part of filename
    :param byteorder: 'native', 'big' or '>', 'little' or '<', default is native
    """
    dn, fn = os.path.split(stem_or_filename)
    nfo = {'stem': fn, 'lend': None, 'bend': None, 'dims': None}
    m = RE_FILENAME.match(fn)
    if m:
        nfo.update(m.groupdict())

    if dirname:
        dn = dirname

    order_code, order_convert = BYTE_ORDER_TABLE.get(byteorder, BYTE_ORDER_NATIVE)
    sfx = order_convert(typename or nfo['lend'] or nfo['bend'])

    if grouping is not None:
        dims = '.'.join(str(x) for x in reversed(grouping))
    else:
        dims = nfo['dims']

    # final filename
    fn = nfo['stem'] + '.' + sfx
    if dims:
        fn += '.' + dims

    # now that we've integrated all the parts, let's get a record dtype out of it
    my_dtype = _dtype_from_path(fn)

    return os.path.join(dn, fn), my_dtype


def _records_in_file(pathname, record_dtype = None):
    """
    return the number of records in a file, requiring that it exists
    :param pathname: flat binary file to look at
    :param record_dtype: numpy dtype used per record
    """
    if record_dtype is None:
        record_dtype = _dtype_from_path(pathname)
    return int(os.stat(pathname).st_size / record_dtype.itemsize)


def memmap(path, mode='r', records=1):
    """
    memory-map a FBF file as a numpy.memmap object
    :param path: file to open or create
    :param mode: 'r', 'w', 'r+', 'c'
    :param records: if new file, number of records to create it with
    :return: numpy memmap object
    """
    dtype = _dtype_from_path(path)
    records = _records_in_file()

    exists = os.path.exists(path)
    writable = 'w' in mode or '+' in mode
    if not exists:
        if not writable:
            raise ValueError('{0} does not exist'.format(path))
        shape = (records,)
    else:
        shape = (_records_in_file(path, dtype), )

    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


class FBF(np.memmap):
    # filename attributes

    def __new__(cls, stemname, typename, grouping=None, dirname=None, byteorder=None, writable=False):
        path, dtype = _construct_from_options(stemname, typename, grouping, dirname, byteorder)
        exists = os.path.exists(path)
        if not exists:
            if not writable:
                raise EnvironmentError('{0} does not exist'.format(path))
            mode = 'w+b'
            shape = (1,)
        else:
            mode ='rb' if not writable else 'r+b'
            shape = (_records_in_file(path, dtype), )

        return np.memmap.__new__(cls, path, dtype=dtype, mode=mode, shape=shape)


    def attach( self, pathname ):
        raise NotImplementedError('deprecated, attachment happens at instantiation')
        return self

    build = attach
    open = attach
    create = attach
    open = attach

    def __len__(self):
        return self.shape[0]


    def open( self, mode='rb' ):
        pass

    def create( self, mode='w+b' ):
        pass

    def close( self ):
        return 0

    def length( self ):
        # return int(os.stat(self.pathname).st_size / self.record_size)
        pass

    def __len__( self ):
        return self.length()
    
    def __getitem__( self, idx ):
        '''obtain records from an FBF file - pass in either an FBF object or an FBF file name.
        Index can be a regular python slice or a 0-based index'''
        length = self.length()
        # if not hasattr(self, 'data') or getattr(self,'mmap_length',0)!=length:
        #     fp = self.fp()
        #     fp.seek(0)
        #     shape = [length] + self.grouping[::-1]
        #     LOG.debug('mapping with shape %r' % shape)
        #     if '+' in self.mode or 'w' in self.mode:
        #         access = mmap.ACCESS_WRITE
        #     else:
        #         access = mmap.ACCESS_READ
        #     self.mmap = mmap.mmap(fp.fileno(), 0, access=access)
        #     self.data = numpy.ndarray( buffer = self.mmap, shape=shape, dtype=self.endian + self.array_type )
        #     self.mmap_length = length
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
    logging.basicConfig(level = logging.DEBUG)
    # doctest.testfile( "numfbf.doctest" )
    unittest.main()
    
if __name__ == '__main__':
    test()
