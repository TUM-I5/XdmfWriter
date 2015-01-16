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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <hdf5.h>

#include <cxxtest/TestSuite.h>

#include "XdmfWriter.h"

class TestXdmfWriter : public CxxTest::TestSuite
{
private:
	int m_rank;

	unsigned int m_cells[3*4];
	double m_vertices[5*3];

	std::vector<const char*> m_varNames;

public:
	void setUp()
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

		srand(m_rank*1000);

		for (unsigned int i = 0; i < 3*4; i++)
			m_cells[i] = rand() % 5;
		for (unsigned int i = 0; i < 5*3; i++)
			m_vertices[i] = rand() / static_cast<double>(RAND_MAX);

		m_varNames.push_back("a");
	}

	// Test append mode
	void testAppend()
	{
		double data[3];

		XdmfWriter writer1(m_rank, "test", m_varNames);
		writer1.init(3, m_cells, 5, m_vertices);

		for (int i = 0; i < 5; i++) {
			setData(i, data);

			writer1.addTimeStep(i);
			writer1.writeData(0, data);
		}

		// Move output files
		if (m_rank == 0) {
			TS_ASSERT_EQUALS(rename("test.h5", "test1.h5"), 0);
			TS_ASSERT_EQUALS(rename("test.xdmf", "test1.xdmf"), 0);
		}
		writer1.close();

		XdmfWriter writer2a(m_rank, "test", m_varNames);
		writer2a.init(3, m_cells, 5, m_vertices);

		for (int i = 0; i < 3; i++) {
			setData(i, data);

			writer2a.addTimeStep(i);
			writer2a.writeData(0, data);
		}
		writer2a.close();

		XdmfWriter writer2b(m_rank, "test", m_varNames, 3);
		writer2b.init(3, m_cells, 5, m_vertices);

		for (int i = 3; i < 5; i++) {
			setData(i, data);

			writer2b.addTimeStep(i);
			writer2b.writeData(0, data);
		}
		writer2b.close();

		// Compare resulting files
		std::ifstream fs1("test1.xdmf");
		std::ifstream fs2("test.xdmf");
		std::istreambuf_iterator<char> eos;
		std::string f1(std::istreambuf_iterator<char>(fs1), eos);
		std::string f2(std::istreambuf_iterator<char>(fs2),	eos);
		TS_ASSERT_EQUALS(f1, f2);

		double data1[3*3*5];
		hid_t h5f = H5Fopen("test1.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
		hid_t h5d = H5Dopen(h5f, "/a", H5P_DEFAULT);
		TS_ASSERT_LESS_THAN_EQUALS(0, H5Dread(h5d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data1));

		double data2[3*3*5];
		h5f = H5Fopen("test.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
		h5d = H5Dopen(h5f, "/a", H5P_DEFAULT);
		TS_ASSERT_LESS_THAN_EQUALS(0, H5Dread(h5d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2));

		TS_ASSERT_EQUALS(memcmp(data1, data2, sizeof(data1)), 0);

		MPI_Barrier(MPI_COMM_WORLD);

		if (m_rank == 0) {
			unlink("test.xdmf");
			unlink("test.h5");
			unlink("test1.xdmf");
			unlink("test1.h5");
		}
	}

private:
	void setData(int step, double* data)
	{
		for (int i = 0; i < 3; i++)
			data[i] = i + 10 * step + 100 * m_rank;
	}
};
