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

#include <cxxtest/TestSuite.h>

#include "ParallelVertexFilter.h"

class TestParallelVertexFilter : public CxxTest::TestSuite
{
public:
	/**
	 * Test with unique vertices
	 */
	void testFilterUnique()
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		double vertices[6*3];
		for (unsigned int i = 0; i < 6*3; i++)
			vertices[i] = RANDOM_VALUES[i+6*3*rank];

		xdmfwriter::internal::ParallelVertexFilter<double> filter;
		filter.filter(6, vertices);

		// Get the global ids
		unsigned long globalIds[6*3];
		MPI_Gather(const_cast<unsigned long*>(filter.globalIds()), 6, MPI_UNSIGNED_LONG,
				globalIds, 6, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
		// There seems the be a bug in allgather in OpenMPI, but we can simulate this
		// with MPI_Bcast
		MPI_Bcast(globalIds, 6*3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

		unsigned int sumVertices = filter.numLocalVertices();
		MPI_Allreduce(MPI_IN_PLACE, &sumVertices, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
		TS_ASSERT_EQUALS(sumVertices, 6*3);

		for (unsigned int i = 0; i < 6*3; i++) {
			TS_ASSERT_EQUALS(globalIds[i], i);
		}
	}

	/**
	 * Test filter with duplicate vertices
	 */
	void testFilterDuplicate()
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		double random_vertices[sizeof(RANDOM_VALUES)];
		memcpy(random_vertices, RANDOM_VALUES, sizeof(RANDOM_VALUES));

		memcpy(&random_vertices[4*3], &random_vertices[8*3], sizeof(double)*3);
		memcpy(&random_vertices[10*3], &random_vertices[8*3], sizeof(double)*3);
		memcpy(&random_vertices[1*3], &random_vertices[14*3], sizeof(double)*3);
		memcpy(&random_vertices[6*3], &random_vertices[15*3], sizeof(double)*3);

		const unsigned int totalVertices = 6*3 - 4;

		double vertices[6*3];
		for (unsigned int i = 0; i < 6*3; i++)
			vertices[i] = random_vertices[i+6*3*rank];

		xdmfwriter::internal::ParallelVertexFilter<double> filter;
		filter.filter(6, vertices);

		// Get the global ids
		unsigned long globalIds[6*3];
		MPI_Gather(const_cast<unsigned long*>(filter.globalIds()), 6, MPI_UNSIGNED_LONG,
				globalIds, 6, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
		// See testFilterUnique why we are not using MPI_Allgather here
		MPI_Bcast(globalIds, 6*3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

		unsigned int sumVertices = filter.numLocalVertices();
		MPI_Allreduce(MPI_IN_PLACE, &sumVertices, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
		TS_ASSERT_EQUALS(sumVertices, totalVertices);

		TS_ASSERT_EQUALS(globalIds[4], globalIds[8]);
		TS_ASSERT_EQUALS(globalIds[10], globalIds[8]);
		TS_ASSERT_EQUALS(globalIds[1], globalIds[14]);
		TS_ASSERT_EQUALS(globalIds[6], globalIds[15]);
	}

	/**
	 * Test filter with similar vertices
	 */
	void testFilterSimilar()
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		double random_vertices[sizeof(RANDOM_VALUES)];
		memcpy(random_vertices, RANDOM_VALUES, sizeof(RANDOM_VALUES));

		memcpy(&random_vertices[4*3], &random_vertices[8*3], sizeof(double)*3);
		reinterpret_cast<uint64_t*>(random_vertices)[4*3] &= ~0xF;
		memcpy(&random_vertices[14*3], &random_vertices[8*3], sizeof(double)*3);
		reinterpret_cast<uint64_t*>(random_vertices)[14*3] |= 0xF;
		memcpy(&random_vertices[1*3], &random_vertices[9*3], sizeof(double)*3);
		reinterpret_cast<uint64_t*>(random_vertices)[9*3+1] |= 0xF;
		memcpy(&random_vertices[6*3], &random_vertices[15*3], sizeof(double)*3);
		reinterpret_cast<uint64_t*>(random_vertices)[15*3+2] |= 0xF;

		const unsigned int totalVertices = 6*3 - 4;

		double vertices[6*3];
		for (unsigned int i = 0; i < 6*3; i++)
			vertices[i] = random_vertices[i+6*3*rank];

		xdmfwriter::internal::ParallelVertexFilter<double> filter;
		filter.filter(6, vertices);

		// Get the global ids
		unsigned long globalIds[6*3];
		MPI_Gather(const_cast<unsigned long*>(filter.globalIds()), 6, MPI_UNSIGNED_LONG,
				globalIds, 6, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
		// See testFilterUnique why we are not using MPI_Allgather here
		MPI_Bcast(globalIds, 6*3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

		unsigned int sumVertices = filter.numLocalVertices();
		MPI_Allreduce(MPI_IN_PLACE, &sumVertices, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
		TS_ASSERT_EQUALS(sumVertices, totalVertices);

		TS_ASSERT_EQUALS(globalIds[4], globalIds[8]);
		TS_ASSERT_EQUALS(globalIds[14], globalIds[8]);
		TS_ASSERT_EQUALS(globalIds[1], globalIds[9]);
		TS_ASSERT_EQUALS(globalIds[6], globalIds[15]);
	}

	const static double RANDOM_VALUES[6*3*3];
};

const double TestParallelVertexFilter::RANDOM_VALUES[6*3*3] = {
		0.1191,
		0.3685,
		0.7968,
		0.791,
		0.154,
		0.8283,
		0.935,
		0.4606,
		0.8961,
		0.931,
		0.8216,
		0.4341,
		0.828,
		0.3907,
		0.1284,
		0.6319,
		0.7405,
		0.7955,
		0.5063,
		0.9716,
		0.3301,
		0.949,
		0.9046,
		0.2141,
		0.8441,
		0.6894,
		0.4295,
		0.7124,
		0.0598,
		0.1737,
		0.4969,
		0.62,
		0.4464,
		0.0551,
		0.049,
		0.3187,
		0.7497,
		0.1793,
		0.307,
		0.5962,
		0.2827,
		0.8266,
		0.8265,
		0.5514,
		0.6731,
		0.7123,
		0.4674,
		0.982,
		0.4975,
		0.5685,
		0.5686,
		0.9724,
		0.2429,
		0.6182
};
