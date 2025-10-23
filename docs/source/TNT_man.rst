
TNT
========================

Reads Tecamg Inc.'s .tnt files and converts to spyctra. Currently uses TNT_reader_lame.py which does not read tables nor the pulse sequence but does read all the user variables. A non-lame version exists that reads everything but... permissions are not fully documented for its public release.


.. py:function:: TNT.read(self, *params)

   Read TNT files and convert to spyctra by interfacing with file_reader.py and tnt_reader_lame.py.

   :param path: The full path to a file or to a set of files minus a numerical suffix. If *path* == None will open a file browser that allows selection of multiple files.
   :type path: str or None
   :param suffixes: The number of files to search for or a list of suffixes to specifically pull. If *path* is not None and *suffixes* is None assumes read is for a single fully specified file.
   :type suffixes: int or iterable[int] or None
   :param options: A string of comma seperated values to suppress output, turn on debugger"
   :type options: str or None
   :return: A new spyctra.
   :rtype: spyctra

