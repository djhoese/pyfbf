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
        raise ValueError('%s%s is not a known suffix' % (bend or '', lend or ''))
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




def filename(stem, dtype, shape=None, multiple_records=False):
    """build a filename, given a stem, element datatype, and record array shape
    filename('mydata', data.dtype, data.shape)
    """
    suffix = _suffix_from_dtype(dtype, shape, multiple_records)
    return stem + '.' + suffix


BYTE_ORDER_LITTLE = ('<', str.lower)
BYTE_ORDER_BIG = ('>', str.upper)
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

    def test_more(self):
        q = np.dtype(('f8', (20, 10)))
        a = np.zeros((5,), dtype=q)
        fn = filename('foo', a, multiple_records=True)
        self.assertEqual('foo.real8.10.20', fn)
        t = _dtype_from_path('/path/to/myfile.REAL8.20.10')
        fn = filename('foo', t, (15,))
        self.assertEqual('foo.REAL8.20.10', fn)
        fn = filename('foo', np.int16, (30, 90))
        self.assertEqual('foo.int2.90.30', fn)




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
    if not os.path.exists(pathname):
        return 0
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
    exists = os.path.exists(path)

    if not exists:
        if 'w' not in mode:
            raise ValueError('{0} does not exist, use "w" mode to create new file'.format(path))
        shape = (records,)
    else:
        shape = (_records_in_file(path, dtype), )

    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


class FBF(object):
    """
    FBF wrapper class which provides writable slicing of FBF files via numpy memory mapping.
    Intended to be backwards compatible with older use patterns, and supplement with out-of-bound implicit file expansion for writable case.
    If you're writing new code and only want read-only, consider just using memfbf.memmap

    """
    path = None  # full path
    dtype = None  # numpy record dtype
    writable = False
    data = None

    def __init__( self, stem_or_filename=None, typename=None, grouping=None, dirname='.', byteorder='native', writable=False ):
        """
        Create an FBF object
        :param stem_or_filename:
        :param typename:
        :param grouping:
        :param dirname:
        :param byteorder:
        :param writable:
        """
        self.writable = writable
        if stem_or_filename is not None:
            self.path, self.dtype = _construct_from_options(stem_or_filename, typename, grouping, dirname, byteorder)
            if os.path.exists(self.path): # then we can open it right now, else we have to wait for a writable slice?
                self.data = memmap(self.path, 'r+' if self.writable else 'r')


    def attach( self, pathname ):
        '''Attach object to existing file on disk
        :param pathname:
        '''
        self.path = pathname
        self.dtype = _dtype_from_path(pathname)
        if os.path.exists(self.path):
            self.data = memmap(self.path, 'r+' if self.writable else 'r')


    def build(self, stem_or_filename, typename, grouping=None, dirname='.', byteorder='native', writable=False ):
        '''build an FBF descriptor object from scratch.
        :param stem_or_filename:
        :param typename:
        :param grouping:
        :param dirname:
        :param byteorder:
        :param writable:
        '''
        self.writable = writable
        self.path, self.dtype = _construct_from_options(stem_or_filename, typename, grouping, dirname, byteorder)


    def __str__(self):
        shape = self.data.shape if self.data is not None else 'empty'
        return "<FBF '{0}' shape {1} dtype {2}>".format(self.path, shape, str(self.dtype))


    def fp(self, mode='rb'):
        raise NotImplementedError('deprecated operation')


    def open( self, mode=None, records=1 ):
        exists = os.path.exists(self.path)
        if mode is None:
            if exists and self.writable:
                mode = 'r+'
            elif exists and not self.writable:
                mode = 'r'
            elif not exists and self.writable:
                mode = 'w+'
            else:
                raise ValueError('{0} does not exist, need mode "w" to create it'.format(self.path))
        self.data = memmap(self.path, mode, records)
        return self


    def create(self, mode='w+', records=1, clobber=True):
        if not clobber and os.path.exists(self.path):
            raise IOError('cannot clobber {0}'.format(self.path))
        self.writable = True
        return self.open(mode, records=records)


    def close( self ):
        if self.data is None:
            return
        self.data.close()
        return 0


    def length( self ):
        if self.data is not None:
            self.data.flush()
            return self.data.shape[0]
        return _records_in_file(self.path, self.dtype)


    def __len__( self ):
        return self.length()


    def _slice_records_required(self, slob, length=None):
        if not isinstance(slob, slice):
            slob = slice(slob)
        def _is_absolute(a,b):
            if a is not None and a<0:
                return False
            if b is not None and b < 0:
                return False
            if a is None and b is None:
                return False
            return True
        if self.writable and _is_absolute(slob.start, slob.stop):
            LOG.debug('absolute and writable')
            start, stop, step = slob.start, slob.stop, slob.step
        else:
            LOG.debug('relative or not writable')
            start, stop, step = slob.indices(length or self.length())
        def _rmax(start, stop):
            if start is None and stop is not None:
                return stop
            if stop is None and start is not None:
                return start + 1
            if start is None and stop is None:
                raise ValueError('cannot compare invalid slice')
            return max(start+1, stop)
        return _rmax(start, stop)


    def _expose_idx(self, idx, reading=True):
        if self.data is None:
            if os.path.exists(self.path):
                self.open()
            elif self.writable:
                records_required = self._slice_records_required(idx)
                self.open(records=records_required)
                self.data.flush()
            else:
                raise IOError('file {0} does not exist'.format(self.path))
        else:
            length = _records_in_file(self.path, self.dtype)
            new_shape = None
            if self.data.shape[0]<length:
                new_shape = (length,) + self.data.shape[1:]

            records_required = self._slice_records_required(idx, length)
            if records_required>length:
                if self.writable:
                    new_shape = (records_required,) + self.data.shape[1:]
                else:
                    raise IndexError('cannot index beyond record {0}'.format(length))

            if new_shape is not None:
                LOG.debug('expanding file to shape {0}'.format(repr(new_shape)))
                self.data.flush()
                self.data.resize(new_shape)

    def __getitem__(self, idx):
        '''obtain records from an FBF file - pass in either an FBF object or an FBF file name.
        Index can be a regular python slice or a 0-based index'''
        self._expose_idx(idx)
        return self.data[idx]

    def __setitem__(self, idx, value):
        self._expose_idx(idx)
        self.data[idx] = value

    @property
    def shape(self):
        return self.data.shape if self.data is not None else None


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


class TestFBF(unittest.TestCase):
    def setUp(self):
        pass

    def test_create(self):
        fbf = FBF('foo', 'real4', [15], writable=True)
        fbf.create(records=3)
        print('shape {0}'.format(repr(fbf.shape)))
        self.assertEqual(fbf.path, './foo.real4.15')
        self.assertEqual(len(fbf), 3)
        a = np.array([np.float32(x) / 2 for x in range(45)], 'f')
        a = a.reshape([3, 15])
        print(fbf[0])
        fbf[0:3] = a[0:3]
        print(fbf[:])
        fbf[3:6] = a[0:3]


def test():
    logging.basicConfig(level = logging.DEBUG)
    unittest.main()


if __name__ == '__main__':
    test()
