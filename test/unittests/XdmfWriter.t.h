/**
 * @file
 *  This file is part of XdmfWriter
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 *
 * @copyright Copyright (c) 2014-2017, Technische Universitaet Muenchen.
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
#include <glob.h>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef USE_HDF
#include <hdf5.h>
#endif // USE_HDF

#include <cxxtest/TestSuite.h>

#include "XdmfWriter.h"

class TestXdmfWriter : public CxxTest::TestSuite
{
private:
	int m_rank;

	unsigned int m_cells[4*4];
	float m_vertices[5*3];

	std::vector<const char*> m_varNames;

public:
	void setUp()
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

		srand(m_rank*1000);

		for (unsigned int i = 0; i < 4*3; i++)
			m_cells[i] = rand() % 5;
		for (unsigned int i = 0; i < 5*3; i++)
			m_vertices[i] = rand() / static_cast<double>(RAND_MAX);

		m_varNames.clear();
		m_varNames.push_back("a");
	}

	void testTriangle()
	{
		float data[5][4];

		m_varNames.push_back("b");

		xdmfwriter::XdmfWriter<xdmfwriter::TRIANGLE, float> writer0(type(), "test");
		writer0.init(m_varNames, std::vector<const char*>());
		writer0.setMesh(4, m_cells, 5, m_vertices);

		for (int i = 0; i < 5; i++) {
			setData(i, data[i]);

			writer0.addTimeStep(i);
			writer0.writeCellData(0, data[i]);
			writer0.writeCellData(1, data[i]);
		}

		writer0.close();

		MPI_Barrier(MPI_COMM_WORLD);

		float dataFile[5][3*4];
		load("test", dataFile[0], sizeof(dataFile));
		for (int i = 0; i < 5; i++) {
			TS_ASSERT_EQUALS(memcmp(data[i], &dataFile[i][4*m_rank], 4*sizeof(float)), 0);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (m_rank == 0) {
			unlink("test.xdmf");
			unlinkDataFiles();
		}
	}

	// Test append mode
	void testAppend()
	{
		float data[4];

		// TODO this currently tests failures only
		xdmfwriter::XdmfWriter<xdmfwriter::TRIANGLE, float> writer0(type(), "test");
		writer0.init(m_varNames, std::vector<const char*>());
		writer0.setMesh(4, m_cells, 5, m_vertices);

		for (int i = 0; i < 5; i++) {
			setData(i, data);

			writer0.addTimeStep(i);
			writer0.writeCellData(0, data);
		}

		writer0.close();

		if (m_rank == 0) {
			unlink("test.xdmf");
			unlinkDataFiles();
		}

		MPI_Barrier(MPI_COMM_WORLD);

		xdmfwriter::XdmfWriter<xdmfwriter::TETRAHEDRON, float> writer1(type(), "test");
		writer1.init(m_varNames, std::vector<const char*>());
		writer1.setMesh(3, m_cells, 5, m_vertices);

		for (int i = 0; i < 5; i++) {
			setData(i, data);

			writer1.addTimeStep(i);
			writer1.writeCellData(0, data);
		}

		writer1.close();

		MPI_Barrier(MPI_COMM_WORLD);

		// Move output files
		if (m_rank == 0) {
#ifdef USE_HDF
			TS_ASSERT_EQUALS(rename("test_cell.h5", "test1_cell.h5"), 0);
			TS_ASSERT_EQUALS(rename("test_vertex.h5", "test1_vertex.h5"), 0);
#else // USE_HDF
			TS_ASSERT_EQUALS(rename("test_cell", "test1_cell"), 0);
			TS_ASSERT_EQUALS(rename("test_vertex", "test1_vertex"), 0);
#endif // USE_HDF
			TS_ASSERT_EQUALS(rename("test.xdmf", "test1.xdmf"), 0);
		}

		xdmfwriter::XdmfWriter<xdmfwriter::TETRAHEDRON, float> writer2a(type(), "test");
		writer2a.init(m_varNames, std::vector<const char*>());
		writer2a.setMesh(3, m_cells, 5, m_vertices);

		for (int i = 0; i < 3; i++) {
			setData(i, data);

			writer2a.addTimeStep(i);
			writer2a.writeCellData(0, data);
		}
		writer2a.close();

		MPI_Barrier(MPI_COMM_WORLD);

		xdmfwriter::XdmfWriter<xdmfwriter::TETRAHEDRON, float> writer2b(type(), "test", 3);
		writer2b.init(m_varNames, std::vector<const char*>());
		writer2b.setMesh(3, m_cells, 5, m_vertices, true);

		for (int i = 3; i < 5; i++) {
			setData(i, data);

			writer2b.addTimeStep(i);
			writer2b.writeCellData(0, data);
		}
		writer2b.close();

		MPI_Barrier(MPI_COMM_WORLD);

		// Compare resulting files
		std::ifstream fs1("test1.xdmf");
		std::ifstream fs2("test.xdmf");
		std::istreambuf_iterator<char> eos;
		std::string f1(std::istreambuf_iterator<char>(fs1), eos);
		std::string f2(std::istreambuf_iterator<char>(fs2), eos);
		TS_ASSERT_EQUALS(f1, f2);

		float data1[3*3*5];
		load("test1", data1, sizeof(data1));

		float data2[3*3*5];
		load("test", data2, sizeof(data2));

		TS_ASSERT_EQUALS(memcmp(data1, data2, sizeof(data1)), 0);

		MPI_Barrier(MPI_COMM_WORLD);

		if (m_rank == 0) {
			unlink("test.xdmf");
			unlinkDataFiles();
			unlink("test1.xdmf");
			unlinkDataFiles("test1");
		}
	}

private:
	void setData(int step, float* data) const
	{
		for (int i = 0; i < 3; i++)
			data[i] = i + 10 * step + 100 * m_rank;
	}

private:
	static xdmfwriter::BackendType type()
	{
#ifdef USE_HDF
		return xdmfwriter::H5;
#else // USE_HDF
		return xdmfwriter::POSIX;
#endif // USE_HDF
	}

	static void unlinkDataFiles(const char* base = "test")
	{
#ifdef USE_HDF
		const char* files[2] = {"_cell.h5", "_vertex.h5"};
		const char* dirs[0] = {};
#else // USE_HDF
		const char* files[5] = {"_cell/mesh0/connect.bin", "_vertex/mesh0/geometry.bin", "_cell/mesh0/partition.bin", "_cell/mesh0/a.bin", "_cell/mesh0/b.bin"};
		const char* dirs[4] = {"_cell/mesh0", "_vertex/mesh0", "_cell", "_vertex"};
#endif // USE_HDF

		const unsigned int numFiles = sizeof(files)/sizeof(const char*);

		for (unsigned int i = 0; i < numFiles; i++) {
			std::string file = std::string(base) + files[i];
			unlink(file.c_str());
		}

		const unsigned int numDirs = sizeof(dirs)/sizeof(const char*);

		for (unsigned int i = 0; i < numDirs; i++) {
			std::string dir = std::string(base) + dirs[i];
			rmdir(dir.c_str());
		}
	}

	static void load(const char* base, float* buffer, size_t size)
	{
#ifdef USE_HDF
		std::string file = std::string(base) + "_cell.h5";
		hid_t h5f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		hid_t h5d = H5Dopen(h5f, "/mesh0/a", H5P_DEFAULT);
		TS_ASSERT_LESS_THAN_EQUALS(0, H5Dread(h5d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer));
#else // USE_HDF
		std::string file = std::string(base) + "_cell/mesh0/a.bin";
		int fh = open64(file.c_str(), O_RDONLY);
		TS_ASSERT_LESS_THAN_EQUALS(0, fh);
		TS_ASSERT_EQUALS(read(fh, buffer, size), size);
		close(fh);
#endif // USE_HDF
	}
};
