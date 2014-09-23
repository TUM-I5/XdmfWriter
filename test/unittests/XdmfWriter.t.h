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
