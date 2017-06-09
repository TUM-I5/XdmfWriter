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

#include <mpi.h>

#include <cxxtest/TestSuite.h>

#include "BufferFilter.h"

class BufferFilter : public CxxTest::TestSuite
{
public:
	void testFilter()
	{
		xdmfwriter::internal::BufferFilter<sizeof(int)> filter;

		int elements0[14] = {0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9};
		unsigned int duplicates0[4] = {2, 5, 9, 10};

		filter.init(14, 4, duplicates0);
		const int* filtered = static_cast<const int*>(filter.filter(elements0));
		for (unsigned int i = 0; i < 10; i++)
			TS_ASSERT_EQUALS(filtered[i], i);

		int elements1[14] = {0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9};
		unsigned int duplicates1[4] = {2, 5, 12, 13};

		filter.init(14, 4, duplicates1);
		filtered = static_cast<const int*>(filter.filter(elements1));
		for (unsigned int i = 0; i < 10; i++)
			TS_ASSERT_EQUALS(filtered[i], i);
	}
};