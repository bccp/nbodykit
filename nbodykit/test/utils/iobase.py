from .. import os
import numpy
     
class IOTestBase(object):
    """
    Base class for testing FileType plugins
    """
    def setUp(self):
        """
        This will handle making/writing the data to
        disk to be loaded into a specific file type
        """
        # the numpy data to write
        self.data = self.make_data()

        # write to disk
        self.path = self.write_to_disk()

        # load the data from disk
        self.file = self.load_data()

    def tearDown(self):
        """
        Remove the temporary file that was used to
        save the data to disk
        """
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_ncol(self):
        """
        Test that the number of columns is right
        """
        self.assertTrue(self.file.ncol == len(self.data.dtype.names))

    def test_size(self):
        """
        Test that the total size of the file is right
        """
        self.assertTrue(self.file.size == len(self.data))

    def test_shape(self):
        """
        Test that the shape of the file is right
        """
        self.assertTrue(self.file.shape == self.data.shape)

    def test_len(self):
        """
        Test that the length of the file is right
        """
        self.assertTrue(len(self.file) == len(self.data))

    def test_iter(self):
        """
        Test that iterating over the file yields the column names
        """
        names = self.data.dtype.names
        self.assertTrue(all(col in names for col in self.file))

    def test_keys(self):
        """
        Test that keys() returns the column names
        """
        names = self.data.dtype.names
        self.assertTrue(sorted(self.file.keys()) == sorted(names))

    def test_integer_index(self):
        """
        Test the indexing of the file using a single integer
        """
        for i in [0, 10, -10]:
            err_msg = "i = %d" %i
            numpy.testing.assert_array_equal(self.file[i], self.data[i], err_msg=err_msg)

    def test_slice(self):
        """
        Test slicing the file
        """
        slices = [slice(0, 10, 1), slice(-10, None, 1)]
        for sl in slices:
            err_msg = "slice = %s" %str(sl)
            numpy.testing.assert_array_equal(self.file[sl], self.data[sl], err_msg=err_msg)

    def test_slice_with_step(self):
        """
        Test slicing the file with a step size > 1
        """
        slices = [slice(0, 10, 2), slice(-10, None, 2)]
        slices += [slice(0, 10, 3), slice(-10, None, 3)]
        for sl in slices:
            err_msg = "slice = %s" %str(sl)
            numpy.testing.assert_array_equal(self.file[sl], self.data[sl], err_msg=err_msg)

    def test_list_of_strings_index(self):
        """
        Test that indexing the file with a list of strings returns a 
        view of only the requested columns
        """
        # grab only the first two columns
        columns = list(self.data.dtype.names[:2])
        ff = self.file[columns]
        
        # dtype and data should only return those of the 2 columns we wanted
        self.assertTrue(ff.dtype == self.data[columns].dtype)
        numpy.testing.assert_array_equal(ff[:], self.data[columns])
        
    def test_string_index(self):
        """
        Test that indexing with a single string index returns a
        numpy array for the requested column (not a structured array)
        """
        # grab a view of only the first column
        column = self.data.dtype.names[0]
        ff = self.file[column]
        
        # this should return the data for the requested column
        self.assertTrue(ff.dtype[column] == self.data.dtype[column])
        numpy.testing.assert_array_equal(ff[:], self.data[column])
        
    
        
        
        
        
            
        
        
            
            
        
        