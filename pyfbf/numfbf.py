'''
$Id$
Flat Binary Format python utilities, recoded for speed using numpy.
Maciek Smuga-Otto <maciek@ssec.wisc.edu> and Ray Garcia
based on the FBF.py library by Ray Garcia <rayg@ssec.wisc.edu>
'''

import logging
import sys
import os
import numpy
import mmap
import string
import re

LOG = logging.getLogger(__name__)

fbf_encodings = dict(
    char1='c', char='c',
    int1='b', uint1='B',
    int2='h', uint2='H',
    int4='i', integer4='i', uint4='I',
    int8='q', uint8='Q',
    real4='f',
    real8='d',
    complex8='F',
    complex16='D'
)

FBF_ENDIAN = {
    '>': string.upper,
    '<': string.lower,
    'big': string.upper,
    'little': string.lower
}


# FIXME: resolve fbf_encodings vs SUFFIX_TO_DTYPE usage

FBF_FLOAT32 = "real4"
FBF_FLOAT64 = "real8"
FBF_INT8 = "int1"
FBF_INT16 = "int2"
FBF_INT32 = "int4"
FBF_INT64 = "int8"
FBF_UINT8 = "uint1"
FBF_UINT16 = "uint2"
FBF_UINT32 = "uint4"
FBF_UINT64 = "uint8"

SUFFIX_TO_DTYPE = {
    FBF_FLOAT32: numpy.float32,
    FBF_FLOAT64: numpy.float64,
    FBF_INT8: numpy.int8,
    FBF_INT16: numpy.int16,
    FBF_INT32: numpy.int32,
    FBF_INT64: numpy.int64,
    FBF_UINT8: numpy.uint8,
    FBF_UINT16: numpy.uint16,
    FBF_UINT32: numpy.uint32,
    FBF_UINT64: numpy.uint64
}

# FIXME avoid this __name__ nonsense
SDTYPE_TO_SUFFIX = dict(
    (v.__name__, k) for (k, v) in SUFFIX_TO_DTYPE.items())   # FUTURE: really would prefer not having a string key

sfx_remap = dict(STA='int1')

# Try to figure out the system's native byte order, defaulting to little-endian if unknown
try:
    BYTEORDER = sys.byteorder
except AttributeError:
    import struct

    testbytes = struct.pack('=l', 0xabcd)
    if struct.pack('<l', 0xabcd) == testbytes:
        BYTEORDER = 'little'
    elif struct.pack('>l', 0xabcd) == testbytes:
        BYTEORDER = 'big'
    else:
        LOG.warning("Unable to identify byte order, defaulting to 'little'.")
        BYTEORDER = 'little'

FBF_ENDIAN['native'] = FBF_ENDIAN['='] = FBF_ENDIAN[BYTEORDER]

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


def _array_product(a):
    LOG.debug("input to _array_product: %s" % repr(a))
    if len(a) == 0:
        return 1
    elif len(a) == 1:
        return a[0]
    else:
        return reduce(lambda x, y: x * y, a) # product of all dimensions


class FBF(object):
    """FBF object.
    Holds lazy file handle, bookkeeping information, mmap object.
    """
    # FUTURE: reduce redundancies here??
    pathname = None
    stemname = None
    dirname = None
    filename = None

    grouping = None
    byteorder = None
    endian = None
    flip_bytes = None
    element_size = None
    record_elements = None
    record_size = None
    array_type = None
    type = None

    pending_flush = None
    file = None
    mode = 'rb'

    data = None
    mmap = None
    mmap_length = None

    def __init__(self, stemname=None, typename=None, grouping=None, dirname='.', byteorder='native', writable=False):
        if not stemname:
            raise FbfWarning('FBF object must be instantiated with at least a filename')
        if not typename:
            filename = stemname # filename given as argument: Attach to existing FBF file on disk
            self.attach(filename, writable=writable)
        else:
            # generate a new FBF object, file will need to be explicitly created.
            if not grouping:
                self.grouping = [1]
            self.build(stemname, typename, grouping, dirname, byteorder, writable)

    def attach(self, pathname, writable=False):
        '''Attach object to existing file on disk'''
        self.pathname = pathname
        self.dirname, self.filename = os.path.split(pathname)
        if not self.dirname: self.dirname = '.'

        parts = self.filename.split('.')

        self.stemname = parts[0]
        fbftype = sfx_remap.get(parts[1].upper(), parts[1])
        self.grouping = [int(x) for x in parts[2:]]
        if not self.grouping:
            self.grouping = [1]

        if fbftype.isupper():
            self.byteorder = 'big'
            self.endian = '>'
        else:
            self.byteorder = 'little'
            self.endian = '<'

        self.flip_bytes = (self.byteorder != BYTEORDER)
        self.element_size = int(re.findall('\d+', fbftype)[0])
        self.record_elements = _array_product(self.grouping)
        self.record_size = self.element_size * self.record_elements
        self.array_type = fbf_encodings[fbftype]
        self.type = fbftype.lower()
        self.pending_flush = False
        self.mode = 'rb' if not writable else 'w+b'

        return self

    def build(self, stemname, typename, grouping=None, dirname='.', byteorder='native', writable=True):
        '''build an FBF descriptor object from scratch.'''

        filename = '%s.%s' % ( stemname, FBF_ENDIAN[byteorder](typename) )
        if grouping and grouping != [1]:
            filename += '.' + '.'.join(str(x) for x in grouping)

        self.attach(os.path.join(dirname, filename))

    def __str__(self):
        return FBF_FMT % vars(self)

    def fp(self, mode=None):
        fob = self.file
        if mode is None:
            mode = self.mode
        if not fob or mode != self.mode:
            if fob is not None:
                fob.close()
            fob = self.file = file(self.pathname, mode)
            self.mode = mode
        return fob

    def open(self, mode='rb'):
        self.file = self.fp(mode)
        return self.file

    def create(self, mode='w+b'):
        self.file = self.fp(mode)
        return self.file

    def close(self):
        if self.file is None:
            return 0
        if self.pending_flush and self.file is not None:
            self.file.flush()
        if self.mmap:
            self.mmap.close()
            self.mmap = None
        self.file.close()
        self.file = None
        return 0

    def length(self):
        """The number of records in the file.
        """
        if self.pending_flush:
            self.file.flush()
            self.pending_flush = False
        return os.stat(self.pathname).st_size / self.record_size

    def __len__(self):
        return self.length()

    def _slice_indices(self, slob, length=None):
        if not isinstance(slob, slice):
            slob = slice(slob)
        return slob.indices(length or self.length())

    def _is_writable(self):
        return '+' in self.mode or 'w' in self.mode

    def __getitem__(self, idx):
        '''obtain records from an FBF file - pass in either an FBF object or an FBF file name.
        Index can be a regular python slice or a 0-based index'''
        if self.pending_flush:
            self.file.flush()

        length = self.length()
        st_size = length * self.record_size
        idxs = self._slice_indices(idx, length)
        st_size_required = (1 + max(idxs)) * self.record_size

        if self._is_writable() and (st_size_required > st_size):
            fp = self.fp()
            fp.seek(st_size_required - 1)
            fp.write('\0')
            fp.flush()
            LOG.debug('extending writable file to size %s by zero filling' % st_size_required)
            length = self.length()

        if self.data is None or (self.record_size * self.mmap_length) < st_size_required:
            fp = self.fp()
            fp.seek(0)
            shape = [length] + self.grouping[::-1]
            LOG.debug('mapping with shape %r' % shape)
            access = mmap.ACCESS_WRITE if self._is_writable() else mmap.ACCESS_READ
            if ((access == mmap.ACCESS_WRITE) and self.flip_bytes):
                raise NotImplementedError(
                    'cannot slice-write an endian-reversed file file currently, use .write() or .block_write()')
            self.mmap = mmap.mmap(fp.fileno(), 0, access=access)
            # FUTURE: look into how much danger we accrue by leaving old mmap un-closed
            self.data = numpy.ndarray(buffer=self.mmap, shape=shape, dtype=self.endian + self.array_type)
            self.mmap_length = length

        #             if self.flip_bytes:
        #                 self.data.dtype = self.data.dtype.newbyteorder()
        # hack: register byte order as swapped in the numpy array without flipping any actual bytes

        #LOG.debug("len: %s grouping: %s" % (self.length(),self.grouping))
        #LOG.debug("data shape: %s returned: %s" % (self.data.shape,self.data[idx].shape))
        return self.data[idx]


def build(stemname, typename, grouping=None, dirname='.', byteorder='native'):
    return FBF(stemname, typename, grouping, dirname, byteorder)


def _slice_ones_to_zeros(start, end, length):
    """convert a one-based start~end inclusive pair to a start:stop zero-based slice, given length of file in records
    """
    # 1~-1 => (0,length)
    # 2~-2 => (1,length-1)
    assert (start != 0)
    start = start - 1 if start > 0 else length + (start + 1)
    end = (None if end == 0
           else length + (end + 1) if end < 0
    else end)
    return slice(start, end)


def read(fbf, start=1, end=0):
    '''legacy API call uses 1-based indexing, so read(1,-1) reads in the whole dataset'''
    if isinstance(fbf, str):
        fbf = FBF(fbf)
        # convert 1-based inclusive [start,end] to 0-based inex [start,end)
    slob = _slice_ones_to_zeros(start, end, fbf.length())
    return fbf[slob][:]


def extract_indices_to_file(inp, indices, out_file):
    "0-based index list is transcribed to a new output file in list order"
    if isinstance(inp, str):
        inp = FBF(inp)
    if isinstance(out_file, str):
        fout = open(out_file, 'wb')
        shouldclose = True
    else:
        shouldclose = False
    for r in indices:
        inp[r].tofile(fout)
    if shouldclose: fout.close()


def write(fbf, idx, data):
    '''write records to an FBF file - pass in either an FBF object or an FBF file name
    The index is a 1-based int (as in Fortran and Matlab). 
    Data must be an appropriately shaped and typed numpy array'''
    # FIXME: can only write in native byte order as far as I can tell.
    # FIXME: coerce data to match dtype of file
    if isinstance(fbf, str): fbf = FBF(fbf)
    if ( _array_product(data.shape) % fbf.record_elements ) != 0:
        raise FbfWarning("data incorrectly shaped for write")

    fbf.file.seek((idx - 1) * fbf.record_size)
    data.tofile(fbf.file)
    fbf.pending_flush = True
    return 1


def block_write(fbf, idx, data):
    return write(fbf, idx, data)


def info(name):
    return FBF(name)


FBF.read = read
FBF.write = write
FBF.block_write = block_write


def test():
    import doctest

    logging.basicConfig(level=logging.DEBUG)
    doctest.testfile("numfbf.doctest")


if __name__ == '__main__':
    test()
