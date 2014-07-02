#! /usr/bin/python

##
# @file
#  This file is part of XdmfWriter
#
# XdmfWriter is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# XdmfWriter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with XdmfWriter.  If not, see <http://www.gnu.org/licenses/>.
#
# @copyright 2014 Technische Universitaet Muenchen
# @author Sebastian Rettenberger <rettenbs@in.tum.de>

hdf5_fortran_prog_src = """
program HDF5_Test
    use hdf5
end program HDF5_Test
"""

hdf5_prog_src_serial = """
#include <hdf5.h>

int main() {
    H5open();
    
    return 0;
}
"""

def CheckHDF5FortranInclude(context):
    context.Message("Checking for Fortran HDF5 module... ")
    ret = context.TryCompile(hdf5_fortran_prog_src, '.f90')
    context.Result(ret)
    
    return ret

def CheckHDF5Linking(context, message):
    context.Message(message+"... ")
    # TODO serial/parallel switch
    ret = context.TryLink(hdf5_prog_src_serial, '.c')
    context.Result(ret)
    
    return ret

def generate(env, **kw):
    conf = env.Configure(custom_tests = {'CheckHDF5FortranInclude' : CheckHDF5FortranInclude,
                                         'CheckHDF5Linking' : CheckHDF5Linking})
    
    if 'required' in kw:
        required = kw['required']
    else:
        required = False
        
    if 'fortran' in kw:
        fortran = kw['fortran']
    else:
        fortran = False
    
    if fortran:
        # Fortran module file
        ret = conf.CheckHDF5FortranInclude()
        if not ret:
            if required:
                print 'Could not find HDF5 for Fortran!'
                env.Exit(1)
            else:
                conf.Finish()
                return
            
        # Fortran library
        ret = conf.CheckLib('hdf5_fortran')
        if not ret:
            if required:
                print 'Could not find HDF5 for Fortran!'
                env.Exit(1)
            else:
                conf.Finish()
                return
        
    ret = [conf.CheckLibWithHeader('hdf5_hl', 'hdf5.h', 'c'), conf.CheckLib('hdf5')]
    if not all(ret):
        if required:
            print 'Could not find HDF5 or zlib!'
            env.Exit(1)
        else:
            conf.Finish()
            return
        
    ret = conf.CheckHDF5Linking("Checking whether shared HDF5 library is used")
    if not ret:
        # Static library, link zlib as well
        ret = conf.CheckLib('z')
        if not ret:
            if required:
                print 'Could not find zlib!'
                env.Exit(1)
            else:
                conf.Finish()
                return
 
        # Try to find all additional libraries
        conf.CheckLib('curl')
        conf.CheckLib('gpfs')
 
        ret = conf.CheckHDF5Linking("Checking whether all HDF5 dependencies are found")
        if not ret:
            if required:
                print 'Could not find all HDF5 dependencies!'
                env.Exit(1)
            else:
                conf.Finish()            
                return
            
    conf.Finish()

def exists(env):
    return True
