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

#include <mpi.h>

#include <cxxtest/TestSuite.h>

#include "BlockBuffer.h"

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

		m_blockBuffer0.init((m_rank+1)*10, MPI_DOUBLE, sizeof(double)*8);

		m_blockBuffer1.init(5, MPI_DOUBLE, sizeof(double)*20);
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
		m_blockBuffer0.exchange(data, out);
		offset[1] = 16; offset[2] = 32;
		for (unsigned int i = 0; i < m_blockBuffer0.count(); i++)
			TS_ASSERT_EQUALS(out[i], offset[m_rank]+i);

		// Second test
		for (int i = 0; i < 5; i++)
			data[i] = i + (m_rank*5);

		m_blockBuffer1.exchange(data, out);
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
};
