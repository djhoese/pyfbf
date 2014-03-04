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

RE_FILENAME = re.compile(
    r'^(?P<stem>\w+)\.(?:(?P<lend>[a-z][a-z0-9]+)|(?P<bend>[A-Z][A-Z0-9]+))(?:\.(?P<dims>[\.0-9]+))?$')
# foo.real4.20.10
#  stem: 'foo'
#  end (bend/lend): 'real4'
#  dims: 20.10


# ref http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing
SUFFIX_TO_TYPESTRING = dict(
    char1='c', char='c',
    int1='i1', uint1='u1', sta='i1',
    int2='i2', uint2='u2',
    int4='i4', uint4='u4',
    int8='i8', uint8='u8',
    real4='f4',
    real8='f8',
    complex8='c8',
    complex16='c16',
)

TYPESTRING_TO_SUFFIX = dict((v, k) for (k, v) in SUFFIX_TO_TYPESTRING.items())


def _dtype_from_regex_groups(lend=None, dims=None, bend=None, **kwargs):
    """
    return a numpy dtype object from a suffix string,
    describing the composition and shape of individual records

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


def suffix_from_dtype(data, shape=None, multiple_records=False):
    """
    return the string file suffix for a file given the data type object representing one record
    :param data: numpy dtype or array
    :param shape: if data is a simple type (numpy.int32 for instance) then this is shape of a record
    :param multiple_records: if data is an array, this determines whether to treat the first dimension as records
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

    assert (typestring[0] in '<|>')
    suffix = TYPESTRING_TO_SUFFIX[typestring[1:]]
    if typestring[0] == '>' or (typestring[0] == '|' and sys.byteorder == 'big'):
        suffix = suffix.upper()
    if shape:
        suffix += '.' + '.'.join(map(str, reversed(shape)))
    return suffix


def filename(stem, dtype, shape=None, multiple_records=False):
    """
    Build a filename, given a stem, element datatype, and record array shape
    filename('mydata', data.dtype, data.shape)
    filename('mydata', data)
    :param stem: variable name without suffix or dimensions
    :param dtype: numpy data type, may or may not include dimensions
    :param shape: if dtype does not include record shape, use this
    :param multiple_records: True if first dtype dimension is the record dimension and should be skipped
    """
    suffix = suffix_from_dtype(dtype, shape, multiple_records)
    return stem + '.' + suffix


BYTE_ORDER_LITTLE = ('<', str.lower)
BYTE_ORDER_BIG = ('>', str.upper)
BYTE_ORDER_NATIVE = BYTE_ORDER_BIG if sys.byteorder == 'big' else BYTE_ORDER_LITTLE
BYTE_ORDER_TABLE = {
    'big': BYTE_ORDER_BIG,
    'little': BYTE_ORDER_LITTLE,
    '>': BYTE_ORDER_BIG,
    '<': BYTE_ORDER_LITTLE
}


def dtype_from_path(pathname):
    """
    pull the numpy dtype out of a pathname
    :param pathname: path to look at
    :return: record dtype
    """
    dn, fn = os.path.split(pathname)
    return _dtype_from_regex_groups(**RE_FILENAME.match(fn).groupdict())


class TestBasics(unittest.TestCase):
    def setUp(self):
        pass

    def test_suffix(self):
        sfx = suffix_from_dtype(np.float64, (10, 20))
        self.assertEqual(sfx, 'real8.20.10')

    def test_array2suffix(self):
        t = np.dtype(('f8', (20, 10)))
        a = np.zeros((5,), dtype=t)
        sfx = suffix_from_dtype(a)
        self.assertEqual(sfx, 'real8.10.20.5')
        sfx = suffix_from_dtype(a, multiple_records=True)
        self.assertEqual(sfx, 'real8.10.20')

    def test_dtype(self):
        m = RE_FILENAME.match('foo.real8.10.20')
        dt = _dtype_from_regex_groups(**m.groupdict())
        a = np.zeros((5,), dt)
        self.assertEqual(a.shape, (5, 20, 10))
        self.assertEqual(a.dtype, np.float64)

    def test_more(self):
        q = np.dtype(('f8', (20, 10)))
        a = np.zeros((5,), dtype=q)
        fn = filename('foo', a, multiple_records=True)
        self.assertEqual('foo.real8.10.20', fn)
        t = dtype_from_path('/path/to/myfile.REAL8.20.10')
        fn = filename('foo', t, (15,))
        self.assertEqual('foo.REAL8.20.10', fn)
        fn = filename('foo', np.int16, (30, 90))
        self.assertEqual('foo.int2.90.30', fn)


def _construct_from_options(stem_or_filename=None, typename=None, grouping=None, dirname=None, byteorder=None):
    """
    Convert a creation specification into a (pathname, record-dtype) pair
    :param stem_or_filename: /path/to/name.real4, name.real4, or name
    :param typename: FBF type name, e.g. 'real4', in the case that stem_or_filename doesn't include it
    :param grouping: tuple of dimensions in access order, e.g. (768,1024) for a 1024wX768h image with rows contiguous
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
    if not dn:
        dn = '.'

    order_code, order_convert = BYTE_ORDER_TABLE.get(byteorder, BYTE_ORDER_NATIVE)
    sfx = typename or nfo['lend'] or nfo['bend']
    if not sfx:
        raise ValueError('{0} is an invalid suffix'.format(sfx))
    sfx = order_convert(sfx)

    if grouping is not None:
        dims = '.'.join(str(x) for x in reversed(grouping))
    else:
        dims = nfo['dims']

    # final filename
    fn = nfo['stem'] + '.' + sfx
    if dims:
        fn += '.' + dims

    # now that we've integrated all the parts, let's get a record dtype out of it
    my_dtype = dtype_from_path(fn)

    return os.path.join(dn, fn), my_dtype


def _records_in_file(pathname, record_dtype=None):
    """
    return the number of records in a file, requiring that it exists
    :param pathname: flat binary file to look at
    :param record_dtype: numpy dtype used per record
    """
    if record_dtype is None:
        record_dtype = dtype_from_path(pathname)
    if not os.path.exists(pathname):
        return 0
    return int(os.stat(pathname).st_size / record_dtype.itemsize)


def memmap(path, mode='r', records=None, dtype=None):
    """
    memory-map a FBF file as a numpy.memmap object
    :param path: file to open or create
    :param mode: 'r', 'w', 'r+', 'c'
    :param records: if new file, number of records to create it with
    :return: numpy memmap object
    """
    dtype = dtype or dtype_from_path(path)
    exists = os.path.exists(path)

    if not exists:
        if 'w' not in mode:
            raise ValueError('{0} does not exist, use "w" mode to create new file'.format(path))
        shape = (records,)
    else:
        shape = (records or _records_in_file(path, dtype), )

    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


class FBF(object):
    """
    FBF wrapper class which provides writable slicing of FBF files via numpy memory mapping.
    Intended to be backwards compatible with older use patterns,
    and supplement with out-of-bound implicit file expansion for writable case.
    If you're writing new code and only want read-only on stably-sized files,
    consider just using memfbf.memmap

    """
    path = None  # full path
    dtype = None  # numpy record dtype
    writable = False # True/False whether we're allowed to write to the file
    data = None # when not None, a numpy memmap object representing some or all the file

    def __init__(self, stem_or_filename=None, typename=None, grouping=None, dirname=None, byteorder='native',
                 writable=False):
        """
        Create an FBF object
        :param stem_or_filename: filename to open, or stem to start with
        :param typename: FBF suffix to use, e.g. 'real4', overriding filename contents
        :param grouping: tuple of array shape for a FBF record, overriding filename
        :param dirname: directory to place the file in, overriding filename
        :param byteorder: 'big' 'little' or 'native', default is native
        :param writable: boolean, whether or not the file should be writable
        """
        self.writable = writable
        if stem_or_filename is not None:
            self.path, self.dtype = _construct_from_options(stem_or_filename, typename, grouping, dirname, byteorder)

    def attach(self, pathname, writable = None):
        """
        Deprecated. Attach an FBF object to a different file
        :param pathname: path of a file to attach or create
        :param writable: boolean, whether the file should be writable or not
        """
        self.path = pathname
        self.dtype = dtype_from_path(pathname)
        if writable is not None:
            self.writable = writable
        if os.path.exists(self.path):
            self.data = memmap(self.path, 'r+' if self.writable else 'r')
        raise DeprecationWarning('FBF.attach is deprecated, use constructor')

    def build(self, stem_or_filename, typename, grouping=None, dirname=None, byteorder='native', writable=False):
        """
        Deprecated. Initialize an FBF after creation.
        :param stem_or_filename: filename to open, or stem to start with
        :param typename: FBF suffix to use, e.g. 'real4', overriding filename contents
        :param grouping: tuple of array shape for a FBF record, overriding filename
        :param dirname: directory to place the file in, overriding filename
        :param byteorder: 'big' 'little' or 'native', default is native
        :param writable: boolean, whether or not the file should be writable
        """
        self.writable = writable
        self.path, self.dtype = _construct_from_options(stem_or_filename, typename, grouping, dirname, byteorder)
        raise DeprecationWarning('FBF.build is deprecated, use constructor')

    def __repr__(self):
        shape = self.data.shape if self.data is not None else 'unopened'
        return "FBF({0}, shape={1}, dtype={2}, records={3})".format(repr(self.path), repr(shape), repr(self.dtype), _records_in_file(self.path, self.dtype))

    def fp(self, mode='rb'):
        raise NotImplementedError('FBF.fp is deprecated')

    def open(self, mode=None, records=None):
        """
        Open the file if it's not already been opened
        :param mode: alternate mode ('r', 'r+', 'w+') if the file is to be re-opened
        :param records: minimum records the file should have if it's being opened as writable
        :raise IOError: if file does not exist and we're not allowed to create it
        """
        exists = os.path.exists(self.path)
        if mode is None:
            if exists and self.writable:
                mode = 'r+'
            elif exists and not self.writable:
                mode = 'r'
            elif not exists and self.writable:
                mode = 'w+'
            else:
                raise IOError('{0} does not exist, need mode "w" to create it'.format(self.path))
        self.data = memmap(self.path, mode, records=records, dtype=self.dtype)

    def create(self, mode='w+', records=0, clobber=True):
        """
        Create the file if it doesn't already exist, reopening if necessary,
        and allocating a minimum number of starting records if necessary.
        :param mode: 'r+', 'r', 'w+', 'c'
        :param records: minimum number of records the file should have
        :param clobber: if false and file already exists, raise an IOError
        """
        if not clobber and os.path.exists(self.path):
            raise IOError('cannot clobber {0}'.format(self.path))
        self.writable = True
        return self.open(mode, records=records)

    def close(self):
        """
        Close the file if it's already open.
        :return:
        """
        if self.data is None:
            return
        self.data.flush()
        self.data = None
        return 0

    def length(self):
        """
        Number of records in the file, as determined by comparing the record size to the file size
        """
        if self.data is not None:
            self.data.flush()
        return _records_in_file(self.path, self.dtype)

    def __len__(self):
        return self.length()

    def _slice_records_required(self, slob, length=None):
        """
        Calculate the number of records that the file would need to contain to access the slice or record index.
        Take into account current file size if relative (negative index) values are used.
        :param slob: record index or slice object
        :param length: file length in records
        :return: required size of the file in records
        :raise ValueError:
        """
        if not isinstance(slob, slice):
            slob = slice(slob)

        def _is_absolute(a, b):
            if a is not None and a < 0:
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
            return max(start + 1, stop)

        return _rmax(start, stop)

    def _expose_idx(self, idx):
        """
        Ensure that the requested index slice is available - or raise IndexError
        Re-maps the whole file if necessary, including expanding read-write files if absolute indices go "off the end" of the file.
        Relative slice indices are considered relative to the file size, not the mapped area size, to ensure consistent behavior.
        :param idx: slice object, absolute or relative (negative indices) to expose as records (outermost dimension)
        :raise IndexError: raised if the file is read-only and the backing file is not sufficiently large
        """
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
            expand_to_records = None
            file_records = _records_in_file(self.path, self.dtype)
            if self.data.shape[0] < file_records:
                expand_to_records = file_records

            records_required = self._slice_records_required(idx, file_records)
            if records_required > file_records:
                if self.writable:
                    expand_to_records = records_required
                else:
                    raise IndexError('cannot index beyond record {0}'.format(file_records))

            if expand_to_records is not None:
                # FUTURE: find a cleaner way than just opening the file, e.g. repair numpy.memmap.resize
                LOG.debug('re-opening file to expose {0} records'.format(expand_to_records))
                self.data.flush()
                self.data = None
                self.open(records=expand_to_records)
                assert (self.data.shape[0] >= expand_to_records)

    def __getitem__(self, idx):
        """
        Access a record slice of the file using memory-mapping.
        :param idx: record number or slice object for records to access
        :return: numpy array view
        """
        self._expose_idx(idx)
        return self.data[idx]

    def __setitem__(self, idx, value):
        """
        Assign data to a writable or copy-on-write memory mapped FBF file.
        :param idx: record number or slice
        :param value: array to assign to the file slice
        """
        self._expose_idx(idx)
        self.data[idx] = value

    @property
    def shape(self):
        """
        Shape of the attached memory map, which may be smaller than the available data in the file if file has grown.
        :return: shape tuple
        """
        if self.data is None:
            self.open()
        return self.data.shape

    @property
    def stemname(self):
        dn,fn = os.path.split(self.path)
        m = RE_FILENAME.match(fn)
        return m.groupdict()['stem']


def build(stemname, typename, grouping=None, dirname='.', byteorder='native', writable=True):
    """
    DEPRECATED: procedural API for creating FBF objects
    """
    return FBF(stemname, typename, grouping, dirname, byteorder, writable)


def _one_based_slice(start, end):
    """
    convert classical 1-based indexing to 0-based slice
    :param start: starting record, with 1 being first record in file
    :param end: inclusive ending record, e.g. -1 for last record inclusive
    :return: zero-based slice object
    """
    if start == end == 0:
        start, end = 1, -1
    idx = start - 1
    if end < -1:
        idx = slice(start - 1, end + 1)
    elif end == -1:
        idx = slice(start - 1, None)  # special case: slice(n,0) is not equivalent
    elif end > 0:
        idx = slice(start - 1, end - 1)
    return idx


def read(fbf, start=0, end=0):
    """
    Legacy API call uses 1-based indexing, so read(1,-1) reads in the whole dataset
    """
    if isinstance(fbf, str):
        LOG.debug('opening file {0} for read'.format(fbf))
        fbf = FBF(fbf)
    idx = _one_based_slice(start, end)
    return fbf[idx]


def extract_indices_to_file(inp, indices, out_file):
    """
    DEPRECATED: 0-based index list is transcribed to a new output file in list order
    """
    if isinstance(inp, str):
        inp = FBF(inp)
    if isinstance(out_file, str):
        out_file = open(out_file, 'wb')
        shouldclose = True
    else:
        shouldclose = False
    for r in indices:
        inp[r].tofile(out_file)
    out_file.flush()
    if shouldclose:
        out_file.close()
    raise DeprecationWarning('FBF.extract_indices_to_file is deprecated')


def write(fbf, idx, data):
    '''DEPRECATED: write records to an FBF file - pass in either an FBF object or an FBF file name
    The index is a 1-based int (as in Fortran and Matlab).
    Data must be an appropriately shaped and typed numpy array'''
    if isinstance(fbf, str):
        fbf = FBF(fbf, writable=True)
        fbf.open(records=data.shape[0])
    fbf[idx] = data
    return 1


def block_write(fbf, idx, data):
    """
    DEPRECATED: block write to a file using one-based indexing
    """
    return write(fbf, idx, data)


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
        LOG.info('shape {0}'.format(repr(fbf.shape)))
        self.assertEqual(fbf.path, './foo.real4.15')
        self.assertEqual(len(fbf), 3)
        a = np.array([np.float32(x) / 2 for x in range(45)], 'f')
        a = a.reshape([3, 15])
        LOG.info(fbf[0])
        fbf[0:3] = a[0:3]
        LOG.info(fbf[:])
        fbf[3:6] = a[0:3]
        os.unlink(fbf.path)

    def test_classic(self):
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

    def test_malformed(self):
        with self.assertRaises(ValueError):
            foo = FBF('any.txt')


def test():
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()


if __name__ == '__main__':
    test()
