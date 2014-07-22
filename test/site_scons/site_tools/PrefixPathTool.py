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

import os

import SCons

def _pathListExists(key, value, env):
    for path in env[key]:
        SCons.Script.PathVariable.PathExists(key, path, env)
        
def _pathToList(value):
    if not value or value == 'none':
        return []
    
    return value.split(os.path.pathsep)

def generate(env, *kw):
    if 'prefixPath' in env:
        
        if 'rpath' in kw:
            rpath = kw['rpath']
        else:
            rpath = True
        
        # Append include/lib and add them to the list if they exist
        incPathes = [p for p in map(lambda p: os.path.join(p, 'include'), env['prefixPath']) if os.path.exists(p)]
        libPathes = [p for p in map(lambda p: os.path.join(p, 'lib64'), env['prefixPath']) if os.path.exists(p)] \
                  + [p for p in map(lambda p: os.path.join(p, 'lib'), env['prefixPath']) if os.path.exists(p)]
                    
        env.AppendUnique(CPPPATH=incPathes)
        env.AppendUnique(LIBPATH=libPathes)
        if rpath:
            env.AppendUnique(RPATH=libPathes)
    
    env['PREFIX_PATH_VARIABLE'] = ('prefixPath',
        'Used when searching for include files, binaries or libraries ( /prefix/path1'+os.path.pathsep+'/prefix/path2 )',
        None,
        _pathListExists,
        _pathToList)

def exists(env):
    return True