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
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 */

#ifndef SCOREP_WRAPPER_H
#define SCOREP_WRAPPER_H

/**
 * Manual instrumentation for Scalasca 2.x with scorep.
 * Override function calls if not compiled with Score-P.
 */
#ifdef SCOREP_USER_ENABLE
#include <scorep/SCOREP_User.h>
#else
#define SCOREP_USER_REGION( name, type )
#define SCOREP_USER_REGION_DEFINE( handle )
#define SCOREP_USER_OA_PHASE_BEGIN( handle, name, type  )
#define SCOREP_USER_OA_PHASE_END( handle )
#define SCOREP_USER_REGION_BEGIN( handle, name, type )
#define SCOREP_USER_REGION_INIT( handle, name, type )
#define SCOREP_USER_REGION_END( handle )
#define SCOREP_USER_REGION_ENTER( handle )
#define SCOREP_USER_FUNC_BEGIN()
#define SCOREP_USER_FUNC_END()
#define SCOREP_GLOBAL_REGION_DEFINE( handle )
#define SCOREP_GLOBAL_REGION_EXTERNAL( handle )
#define SCOREP_USER_PARAMETER_INT64( name, value )
#define SCOREP_USER_PARAMETER_UINT64( name, value )
#define SCOREP_USER_PARAMETER_STRING( name, value )
#define SCOREP_USER_METRIC_GLOBAL( metricHandle )
#define SCOREP_USER_METRIC_EXTERNAL( metricHandle )
#define SCOREP_USER_METRIC_LOCAL( metricHandle )
#define SCOREP_USER_METRIC_INIT( metricHandle, name, unit, type, context )
#define SCOREP_USER_METRIC_INT64( metricHandle, value )
#define SCOREP_USER_METRIC_UINT64( metricHandle, value )
#define SCOREP_USER_METRIC_DOUBLE( metricHandle, value )
#define SCOREP_RECORDING_ON()
#define SCOREP_RECORDING_OFF()
#define SCOREP_RECORDING_IS_ON() 0
#endif

#endif // SCOREP_WRAPPER_H
