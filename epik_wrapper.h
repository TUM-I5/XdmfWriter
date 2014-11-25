/**
 * @file
 *  This file is part of XdmfWriter
 *
 * XdmfWriter is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * XdmfWriter is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with XdmfWriter.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @copyright 2014 Technische Universitaet Muenchen
 * @author Sebastian Rettenberger <rettenbs@in.tum.de>
 */

#ifndef EPIK_WRAPPER_H
#define EPIK_WRAPPER_H

/**
 * Manual instrumentation for Scalasca with epik.
 * Override function calls if not compiled with EPIK.
 */
#ifdef EPIK
#include "epik_user.h"
#else
#define EPIK_FUNC_REG(str)
#define EPIK_FUNC_START()
#define EPIK_FUNC_END()
#define EPIK_USER_REG(id,str)
#define EPIK_USER_START(id)
#define EPIK_USER_END(id)
#define EPIK_TRACER(str)
#endif

#endif EPIK_WRAPPER_H
