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

#ifndef BLOCK_BUFFER_SERIAL_H
#define BLOCK_BUFFER_SERIAL_H

/**
 * Dummy implementation for serial code. Only implements some functions.
 * Use this class to avoid some <code>#ifdef USE_MPI<code>/<code>#endif</code> constructs.
 */
class BlockBuffer
{
public:
	virtual ~BlockBuffer()
	{
	}

	/**
	 * @return Always <code>false</code>
	 */
	bool isInitialized() const
	{
		return false;
	}

	/**
	 * @return Always 1
	 */
	unsigned int count() const
	{
		return 1;
	}
};

#endif // BLOCK_BUFFER_SERIAL_H
