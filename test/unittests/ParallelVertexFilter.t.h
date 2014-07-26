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

		ParallelVertexFilter filter;
		filter.filter(6, vertices);

		// Get the global ids
		unsigned long globalIds[6*3];
		MPI_Gather(const_cast<unsigned long*>(filter.globalIds()), 6, MPI_UNSIGNED_LONG,
				globalIds, 6, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
		// There seems the be a bug in allgather in OpenMPI, but we can simulate this
		// with MPI_Bcast
		MPI_Bcast(globalIds, 6*3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			// Sort the vertices according to the global id array
			double sortedVertices[6*3*3];
			memset(sortedVertices, 0, sizeof(double)*6*3*3);
			for (unsigned int i = 0; i < 6*3; i++) {
				memcpy(&sortedVertices[globalIds[i]*3], &RANDOM_VALUES[i*3], sizeof(double)*3);
			}

			for (unsigned int i = 1; i < 6*3; i++) {
				//logInfo() << rank << sortedVertices[(i-1)*3] << sortedVertices[i*3];
				TS_ASSERT(lessEqual(&sortedVertices[(i-1)*3], &sortedVertices[i*3]));
			}
		}

		unsigned int sumVertices = filter.numLocalVertices();
		MPI_Allreduce(MPI_IN_PLACE, &sumVertices, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
		TS_ASSERT_EQUALS(sumVertices, 6*3);
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

		ParallelVertexFilter filter;
		filter.filter(6, vertices);

		// Get the global ids
		unsigned long globalIds[6*3];
		MPI_Gather(const_cast<unsigned long*>(filter.globalIds()), 6, MPI_UNSIGNED_LONG,
				globalIds, 6, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
		// See testFilterUnique why we are not using MPI_Allgather here
		MPI_Bcast(globalIds, 6*3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

		// Sort the vertices according to the global id array
		double sortedVertices[6*3*3];
		memset(sortedVertices, 0, sizeof(double)*6*3*3);
		for (unsigned int i = 0; i < 6*3; i++) {
			memcpy(&sortedVertices[globalIds[i]*3], &random_vertices[i*3], sizeof(double)*3);
		}

		for (unsigned int i = 1; i < totalVertices; i++) {
			//logInfo() << rank << sortedVertices[(i-1)*3] << sortedVertices[i*3];
			TS_ASSERT(lessEqual(&sortedVertices[(i-1)*3], &sortedVertices[i*3]));
		}

		unsigned int sumVertices = filter.numLocalVertices();
		MPI_Allreduce(MPI_IN_PLACE, &sumVertices, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
		TS_ASSERT_EQUALS(sumVertices, totalVertices);
	}

private:
	static bool lessEqual(const double* vertexA, const double* vertexB)
	{
		return (vertexA[0] < vertexB[0])
				|| (vertexA[0] == vertexB[0] && vertexA[1] < vertexB[1])
				|| (vertexA[0] == vertexB[0] && vertexA[1] == vertexB[1]
					&& vertexA[2] < vertexB[2]);
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