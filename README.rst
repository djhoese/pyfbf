pyfbf
=====

What is PyFBF?
--------------

Python library for working with flat binary files (FBF).

This library is primarily used by software developers at the Space Science and Engineering Center (SSEC) at the University of Wisconsin - Madison.

Installation
------------

To install PyFBF run ``python setup.py install``.

FBF Naming Scheme
-----------------

    <stem>.<data type>.<dim...>

``stem`` can be any identifying string and cannot contain a period character.

``data type`` can be any one of the following (numpy equivalent in parantheses):

 - int1: 8-bit signed integer (int8)
 - int2: 16-bit signed integer (int16)
 - int4: 32-bit signed integer (int32)
 - int8: 64-bit signed integer (int64)
 - uint1: 8-bit unsigned integer (uint8)
 - uint2: 16-bit unsigned integer (uint16)
 - uint4: 32-bit unsigned integer (uint32)
 - uint8: 64-bit unsigned integer (uint64)
 - real4: 32-bit float (float32)
 - real8: 64-bit float (float64)
 - complex8: 64-bit complex number, represented as 2 32-bit floats, real and imaginary (complex64)
 - complex16: 128-bit complex number, represented as 2 64-bit floats, real and imaginary (complex128)
 - sta: Special status file type of 8-bit integers; 0=untouched, -1=bad data, 1=good data (int1)

An UPPERCASE ``data type`` (.REAL4) represents big-endian data, lowercase represent little-endian data (.real4).

Any remaining information in the flat binary filename represent the dimensionality of the data. Data is ordered in the order specified in the filename. Binary bytes are first grouped by their data type (32-bit floats for example) then by the next dimension in the filename and so on. An unspecified 'record' dimension is also created by dividing the size of the file by the number of elements so far. In software the record dimension starts counting from 1; 0 is a non-record.

Example:
A filename of ``test_data.real4.128.256`` with a file size of 327680 bytes would have 10 records ( 327680 / 4 / 128 / 256 = 10). When loaded into python this file will be represented as a numpy array with the shape of (10, 256, 128) and would have a numpy data type of "float32".

Basic Usage
-----------

Load binary data files from a directory (workspace) as numpy arrays.

    from pyfbf import Workspace
    w = Workspace('/mydata')
    time_data = w.time[:]
    air_temp_data = w.air_temp[:]
    
Where the time data is read from a file named ``time.real4`` and the air temperature data from a file named ``air_temp.real4``.



