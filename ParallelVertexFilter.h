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

#ifndef PARALLEL_VERTEX_FILTER_H
#define PARALLEL_VERTEX_FILTER_H

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdint.h> // TODO switch to cstdint as soon as all builds enable C++11
#include <vector>

#include "utils/logger.h"

#include "epik_wrapper.h"

/**
 * Filters duplicate vertices in parallel
 */
class ParallelVertexFilter
{
private:
	/**
	 * Compares 3D-vertex indices according to the vertices
	 */
	class IndexedVertexComparator
	{
	private:
		const double *m_vertices;

	public:
		IndexedVertexComparator(const double *vertices)
			: m_vertices(vertices)
		{
		}

		bool operator() (unsigned int i, unsigned int j)
		{
			i *= 3;
			j *= 3;

			return (m_vertices[i] < m_vertices[j])
					|| (m_vertices[i] == m_vertices[j] && m_vertices[i+1] < m_vertices[j+1])
					|| (m_vertices[i] == m_vertices[j] && m_vertices[i+1] == m_vertices[j+1]
						&& m_vertices[i+2] < m_vertices[j+2]);
		}
	};

private:
	/** The communicator we use */
	MPI_Comm m_comm;

	/** Our rank */
	int m_rank;

	/** #Processes */
	int m_numProcs;

	/** Global id after filtering */
	unsigned long *m_globalIds;

	/** Number of local vertices after filtering */
	unsigned int m_numLocalVertices;

	/** Local vertices after filtering */
	double *m_localVertices;

public:
	ParallelVertexFilter(MPI_Comm comm = MPI_COMM_WORLD)
		: m_comm(comm), m_globalIds(0L), m_numLocalVertices(0), m_localVertices(0L)
	{
		MPI_Comm_rank(comm, &m_rank);
		MPI_Comm_size(comm, &m_numProcs);

		if (vertexType == MPI_DATATYPE_NULL) {
			MPI_Type_contiguous(3, MPI_DOUBLE, &vertexType);
			MPI_Type_commit(&vertexType);
		}
	}

	virtual ~ParallelVertexFilter()
	{
		delete [] m_globalIds;
		delete [] m_localVertices;
	}

	/**
	 * @param vertices Vertices that should be filtered, must have the size <code>numVertices * 3</code>
	 */
	void filter(unsigned int numVertices, const double *vertices)
	{
		EPIK_TRACER("ParallelVertexFilter_Filter");

		// Chop the last 4 bits to avoid numerical errors
		double *roundVertices = new double[numVertices*3];
		removeRoundError(vertices, numVertices*3, roundVertices);

		// Create indices and sort them locally
		unsigned int *sortIndices = new unsigned int[numVertices];
		createSortedIndices(roundVertices, numVertices, sortIndices);

		// Select BUCKETS_PER_RANK-1 splitter elements
		double localSplitters[BUCKETS_PER_RANK-1];
#if 0 // Use omp only if we create a larger amount of buckets
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
#endif
		for (int i = 0; i < BUCKETS_PER_RANK-1; i++) {
			unsigned long vrtxIndex = static_cast<unsigned long>(i)
					* static_cast<unsigned long>(numVertices)
					/ static_cast<unsigned long>(BUCKETS_PER_RANK-1);
			assert(vrtxIndex < numVertices);

			localSplitters[i] = roundVertices[sortIndices[vrtxIndex]*3];
		}

		// Collect all splitter elements on rank 0
		double *allSplitters = 0L;

		if (m_rank == 0)
			allSplitters = new double[m_numProcs * (BUCKETS_PER_RANK-1)];

		MPI_Gather(localSplitters, BUCKETS_PER_RANK-1, MPI_DOUBLE,
				allSplitters, BUCKETS_PER_RANK-1, MPI_DOUBLE,
				0, m_comm);

		// Sort splitter elements
		if (m_rank == 0)
			std::sort(allSplitters, allSplitters + (m_numProcs * (BUCKETS_PER_RANK-1)));

		// Distribute splitter to all processes
		double *splitters = new double[m_numProcs-1];

		if (m_rank == 0) {
#ifdef _OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (int i = 0; i < m_numProcs-1; i++) {
				unsigned long spltIndex = (i+1) * (BUCKETS_PER_RANK-1);
				assert(spltIndex < static_cast<unsigned int>(m_numProcs * (BUCKETS_PER_RANK-1)));

				splitters[i] = allSplitters[spltIndex];
			}
		}

		MPI_Bcast(splitters, m_numProcs-1, MPI_DOUBLE, 0, m_comm);

		delete [] allSplitters;

		// Determine the bucket for each vertex
		unsigned int *bucket = new unsigned int[numVertices];

#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++) {
			double* ub = std::upper_bound(splitters, splitters+m_numProcs-1, roundVertices[i*3]);

			bucket[i] = ub-splitters;
		}

		delete [] roundVertices;
		delete [] splitters;

		// Determine the (local and total) bucket size
		int *bucketSize = new int[m_numProcs];
		memset(bucketSize, 0, sizeof(int)*m_numProcs);
		for (unsigned int i = 0; i < numVertices; i++)
			bucketSize[bucket[i]]++;

		delete [] bucket;

		// Tell all processes what we are going to send them
		int *recvSize = new int[m_numProcs];

		MPI_Alltoall(bucketSize, 1, MPI_INT, recvSize, 1, MPI_INT, m_comm);

		unsigned int numSortVertices = 0;
#ifdef _OPENMP
		#pragma omp parallel for schedule(static) reduction(+: numSortVertices)
#endif
		for (int i = 0; i < m_numProcs; i++)
			numSortVertices += recvSize[i];

		// Create sorted send buffer
		double *sendVertices = new double[3 * numVertices];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++) {
			memcpy(&sendVertices[i*3], &vertices[sortIndices[i]*3], sizeof(double)*3);
		}

		// Allocate buffer for the vertices and exchange them
		double *sortVertices = new double[3 * numSortVertices];

		int *sDispls = new int[m_numProcs];
		int *rDispls = new int[m_numProcs];
		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < m_numProcs; i++) {
			sDispls[i] = sDispls[i-1] + bucketSize[i-1];
			rDispls[i] = rDispls[i-1] + recvSize[i-1];
		}
		MPI_Alltoallv(sendVertices, bucketSize, sDispls, vertexType, sortVertices, recvSize, rDispls, vertexType, m_comm);

		delete [] sendVertices;

		// Chop the last 4 bits to avoid numerical errors
		roundVertices = new double[numSortVertices*3];
		removeRoundError(sortVertices, numSortVertices*3, roundVertices);

		// Create indices and sort them (such that the vertices are sorted)
		unsigned int *sortSortIndices = new unsigned int[numSortVertices];
		createSortedIndices(roundVertices, numSortVertices, sortSortIndices);

		delete [] roundVertices;

		// Initialize the global ids we send back to the other processors
		unsigned long *gids = new unsigned long[numSortVertices];

		gids[sortSortIndices[0]] = 0;
		for (unsigned int i = 1; i < numSortVertices; i++) {
			if (equals(&sortVertices[sortSortIndices[i-1]*3], &sortVertices[sortSortIndices[i]*3]))
				gids[sortSortIndices[i]] = gids[sortSortIndices[i-1]];
			else
				gids[sortSortIndices[i]] = gids[sortSortIndices[i-1]] + 1;
		}

		// Create the local vertices list
		m_numLocalVertices = gids[sortSortIndices[numSortVertices-1]] + 1;
		delete [] m_localVertices;
		m_localVertices = new double[m_numLocalVertices * 3];
		for (unsigned int i = 0; i < numSortVertices; i++)
			memcpy(&m_localVertices[gids[i]*3], &sortVertices[i*3], sizeof(double)*3);

		delete [] sortVertices;

		// Get the vertices offset
		unsigned int offset = m_numLocalVertices;
		MPI_Scan(MPI_IN_PLACE, &offset, 1, MPI_UNSIGNED, MPI_SUM, m_comm);
		offset -= m_numLocalVertices;

		// Add offset to the global ids
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numSortVertices; i++)
			gids[i] += offset;

		// Send result back
		unsigned long *globalIds = new unsigned long[numVertices];
		MPI_Alltoallv(gids, recvSize, rDispls, MPI_UNSIGNED_LONG,
				globalIds, bucketSize, sDispls, MPI_UNSIGNED_LONG, m_comm);

		delete [] bucketSize;
		delete [] recvSize;
		delete [] sDispls;
		delete [] rDispls;
		delete [] gids;

		// Assign the global ids to the correct vertices
		delete [] m_globalIds;
		m_globalIds = new unsigned long[numVertices];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++)
			m_globalIds[sortIndices[i]] = globalIds[i];

		delete [] sortIndices;
		delete [] globalIds;
	}

	/**
	 * @return The list of the global identifiers after filtering
	 */
	const unsigned long* globalIds()
	{
		return m_globalIds;
	}

	/**
	 * @return Number of vertices this process is responsible for after filtering
	 */
	unsigned int numLocalVertices()
	{
		return m_numLocalVertices;
	}

	/**
	 * @return The list of vertices this process is responsible for after filtering
	 */
	const double* localVertices()
	{
		return m_localVertices;
	}

private:
	/**
	 * Removes round errors of double values by setting the last 4 bits
	 * (of the significand) to zero.
	 *
	 * @warning Only works if <code>value</code> ist not nan or infinity
	 * @todo This should work for arbitrary precision
	 */
	static double removeRoundError(double value)
	{
		static const uint64_t mask = ~0xF;

		union FloatUnion {
			double f;
			uint64_t bits;
		};

		FloatUnion result;
		result.f = value;

		result.bits &= mask;

		return result.f;
	}

	/**
	 * Removes the round errors using {@link removeRoundError(double)}
	 *
	 * @param values The list of floating point values
	 * @param count Number of values
	 * @param[out] roundValues The list of rounded values
	 *  (the caller is responsible for allocating the memory)
	 */
	static void removeRoundError(const double *values, unsigned int count, double* roundValues)
	{
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < count; i++)
			roundValues[i] = removeRoundError(values[i]);
	}

	/**
	 * Creates the list of sorted indices for the vertices.
	 * The caller is responsible for allocating the memory.
	 */
	static void createSortedIndices(const double *vertices, unsigned int numVertices,
			unsigned int *sortedIndices)
	{

#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++)
			sortedIndices[i] = i;

		IndexedVertexComparator comparator(vertices);
		std::sort(sortedIndices, sortedIndices+numVertices, comparator);
	}

	/**
	 * Compares to vertices for equality
	 * Assumes that the rounding errors are removed.
	 */
	static bool equals(const double* vertexA, const double* vertexB)
	{
		return vertexA[0] == vertexB[0]
		       && vertexA[1] == vertexB[1]
		       && vertexA[2] == vertexB[2];
	}

	/** MPI data type consisting of three doubles */
	static MPI_Datatype vertexType;

	/** The total buckets we create is <code>BUCKETS_PER_RANK * numProcs</code> */
	const static int BUCKETS_PER_RANK = 8;
};

#endif // PARALLEL_VERTEX_FILTER_H
