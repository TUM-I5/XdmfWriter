/**
 * @file
 *  This file is part of XdmfWriter
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 *
 * @copyright Copyright (c) 2017, Technische Universitaet Muenchen.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from this
 *     software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef XDMFWRITER_BUFFERFILTER_H
#define XDMFWRITER_BUFFERFILTER_H

#include <cassert>
#include <cstring>

#include "utils/logger.h"

namespace xdmfwriter
{

namespace internal
{

/**
 * Filters a buffer by removing elements from a given index list
 *
 * @tparam S The size of an element (in bytes)
 */
template<unsigned int S>
class BufferFilter
{
private:
	/** The old number of vertices */
	unsigned int m_numElements;

	/** The number of duplicates */
	unsigned int m_numDuplicates;

	/** The list of duplicates */
	const unsigned int* m_duplicates;

	/** The buffer to collect the data */
	char* m_buffer;
public:
	BufferFilter()
		: m_numElements(0), m_numDuplicates(0),
		m_duplicates(0L), m_buffer(0L)
	{ }

	virtual ~BufferFilter()
	{
		delete [] m_buffer;
	}

	void init(unsigned int numElements, unsigned int numDuplicates, const unsigned int* duplicates)
	{
		m_numElements = numElements;
		m_numDuplicates = numDuplicates;
		m_duplicates = duplicates;

		delete [] m_buffer;
		m_buffer = new char[S * (numElements - numDuplicates)];
	}

	const void* filter(const void* data)
	{
		unsigned int j = 0; // counting the result index
		for (unsigned int i = 0; i < m_numElements; i++) {
			if ((i-j) >= m_numDuplicates || m_duplicates[i-j] != i) {
				assert(j < m_numElements - m_numDuplicates);
				memcpy(&m_buffer[j*S], &static_cast<const char*>(data)[i*S], S);
				j++;
			}
		}

		return m_buffer;
	}
};

}

}

#endif // XDMFWRITER_BUFFERFILTER_H