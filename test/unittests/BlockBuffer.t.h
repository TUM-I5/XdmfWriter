/**
 * @file
 *  This file is part of XdmfWriter
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 *
 * @copyright Copyright (c) 2014-2015, Technische Universitaet Muenchen.
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

#include "backends/BlockBuffer.h"

class TestBlockBuffer : public CxxTest::TestSuite
{
private:
	int m_rank;

	BlockBuffer m_blockBuffer0;
	BlockBuffer m_blockBuffer1;

public:
	void setUp()
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

		m_blockBuffer0.init(MPI_COMM_WORLD, (m_rank+1)*10, sizeof(double), sizeof(double)*8);

		m_blockBuffer1.init(MPI_COMM_WORLD, 5, sizeof(double), sizeof(double)*20);
	}

	void testIsInitialized()
	{
		BlockBuffer b;
		TS_ASSERT(!b.isInitialized());

		TS_ASSERT(m_blockBuffer0.isInitialized());
	}

	void testCount()
	{
		unsigned int count0[] = {16, 16, 28};
		TS_ASSERT_EQUALS(m_blockBuffer0.count(), count0[m_rank]);

		unsigned int count1[] = {15, 0, 0};
		TS_ASSERT_EQUALS(m_blockBuffer1.count(), count1[m_rank]);
	}

	void testExchange()
	{
		// First test
		unsigned int offset[] = {0, 10, 30};
		double data[30];
		for (int i = 0; i < (m_rank+1)*10; i++)
			data[i] = i + offset[m_rank];

		double out[28];
		m_blockBuffer0.exchange(data, MPI_DOUBLE, 1, out);
		offset[1] = 16; offset[2] = 32;
		for (unsigned int i = 0; i < m_blockBuffer0.count(); i++)
			TS_ASSERT_EQUALS(out[i], offset[m_rank]+i);

		// Second test
		for (int i = 0; i < 5; i++)
			data[i] = i + (m_rank*5);

		m_blockBuffer1.exchange(data, MPI_DOUBLE, 1, out);
		for (unsigned int i = 0; i < m_blockBuffer1.count(); i++)
			TS_ASSERT_EQUALS(out[i], i);

		// Test advanced exchange
		offset[1] = 20; offset[2] = 60;
		short data2[60];
		for (int i = 0; i < (m_rank+1)*20; i++)
			data2[i] = i + offset[m_rank];

		short out2[56];
		m_blockBuffer0.exchange(data2, MPI_SHORT, 2, out2);
		offset[1] = 32; offset[2] = 64;
		for (unsigned int i = 0; i < m_blockBuffer0.count()*2; i++)
			TS_ASSERT_EQUALS(out2[i], offset[m_rank]+i);
	}

	void testExchangeAny()
	{
		// First test
		float data[10];
		for (unsigned int i = 0; i < 10; i++)
			data[i] = i+m_rank*10;

		unsigned int count;
		float *out = m_blockBuffer0.exchangeAny(data, MPI_FLOAT, 2, 5, count);
		TS_ASSERT_EQUALS(count, 5);
		for (unsigned int i = 0; i < 10; i++)
			TS_ASSERT_EQUALS(out[i], i+m_rank*10);
		BlockBuffer::free(out);

		// Second test
		out = m_blockBuffer1.exchangeAny(data, MPI_FLOAT, 2, 5, count);
		TS_ASSERT_EQUALS(count, (m_rank == 0 ? 15 : 0));
		if (m_rank == 0) {
			for (unsigned int i = 0; i < 30; i++)
				TS_ASSERT_EQUALS(out[i], i);
		}
		BlockBuffer::free(out);
	}
};
