"""utility

This module defines common variables and functions.

Todo:
    * output paths for visualisation
    * output paths for firedrake
"""
import os
import glob
import errno
import sys, getopt
from firedrake import COMM_WORLD

class Workspace(object):
    """The summary line for a class docstring should fit on one line.
    """
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def output_name(self, filename='', sub_dir=''):
        """Generates output filename/directory.

        Returns:
            base_directory/sub_dir/filename
        """
        _dirname = self.base_directory if sub_dir=='' else self.base_directory + '/' + sub_dir
        if not os.path.exists(_dirname):
            try:
                os.makedirs(_dirname, 0o755)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        return _dirname if filename=='' else _dirname + '/' + filename

    def clear_directory(self, sub_dir=''):
        """Removes all files from base_directory/sub_directory.
        """
        _dirpath = self.base_directory if sub_dir=='' else self.base_directory + '/' + sub_dir
        import shutil
        for filename in os.listdir(_dirpath):
            filepath = os.path.join(_dirpath, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)

    def list_all_h5_files(self, sub_dir='', prefix=''):
        """
        return a list of all h5 files in workspace/sub_dir
        """
        _dirpath = self.base_directory if sub_dir=='' else self.base_directory + '/' + sub_dir
        fileset = [filename for filename in glob.glob(_dirpath + "/{}*.h5".format(prefix), recursive=True)]
        return fileset

    def sub_dir(self, sub_dir_folder_name=''):
        """
        returns subfolder dir with "/" appended
        """
        _return_value = self.output_name('',sub_dir_folder_name)
        # print(_return_value)

        return _return_value + '/'
        #_return_value = self.base_directory + '/' + sub_dir_folder_name
        #return self.base_directory if sub_dir_folder_name=='' else _return_value 

    
def commandline_parser(argv):
    """Commandline parser for TQG solver.

    Returns:
       A dictionary object that contains values for
       time, resolution, time_step and dump_freq
    """
    T = 0.
    res = 64
    dt = 0.01
    dump_freq = 1
    alpha=None
    do_save_data=False
    do_save_visual=True
    do_save_spectrum=False
    
    try:
        options, remainder = getopt.gnu_getopt(
                argv,
                'T:n:t:f:a:',
                ['time=',
                 'resolution=',
                 'time_step=',
                 'dump_freq=',
                 'alpha=',
                 'nsave_data',
                 'nsave_visual',
                 'save_data',
                 'save_visual',
                 'save_spectrum',
                 ])
        # print(options)
    except getopt.GetoptError as err:
        print('Error:', err)
        sys.exit(1)

    for opt, arg in options:
        if opt in ('-T', '--time'):
            T = float(arg)
        elif opt in ('-n', '--resolution'):
            res = int(arg)
        elif opt in ('-t', '--time_step'):
            dt = float(arg)
        elif opt in ('-f', '--dump_freq'):
            dump_freq = int(arg)
        elif opt in ('-a', '--alpha'):
            alpha = float(arg)
        elif opt in ('--nsave_data'):
            do_save_data=False
        elif opt in ('--nsave_visual'):
            do_save_visual=False
        elif opt in ('--save_data'):
            do_save_data=True
        elif opt in ('--save_visual'):
            do_save_visual=True
        elif opt in ('--save_spectrum'):
            do_save_spectrum=True
        else:
            assert False, "unhandled option"

    return {'time': T, 'resolution': res, 'time_step': dt, 'dump_freq': dump_freq, 'alpha': alpha, 'do_save_data': do_save_data, 'do_save_visual': do_save_visual, 'do_save_spectrum': do_save_spectrum} 

