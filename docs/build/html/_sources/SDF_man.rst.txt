
SDF
========================

Reads Stelar S.R.L.'s .sdf files and converts to a list of spyctra. Reads all data and all metadata.


.. py:function:: read(self, *params)

   Read the selected experiments from path and convert to a list of spyctra.

   :param path: The full path to a file.
   :type path: str
   :param find: The number of experiments to pull from the file.
   :type find: int
   :param options: A string of comma seperated values to suppress output, turn on debugger, etc.
   :type options: str or None
   :return: A list of spyctra.
   :rtype: list[spyctra]

