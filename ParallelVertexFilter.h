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

#ifndef XDMFWRITER_PARALLELVERTEXFILTER_H
#define XDMFWRITER_PARALLELVERTEXFILTER_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <vector>

#include "utils/logger.h"

#include "FloatUnion.h"
#include "IndexSort.h"
#include "scorep_wrapper.h"

namespace xdmfwriter
{

namespace internal
{

/**
 * Filters duplicate vertices in parallel
 */
template<typename T>
class ParallelVertexFilter
{
private:
#ifdef USE_MPI
	/** The communicator we use */
	MPI_Comm m_comm;
#endif // USE_MPI

	/** Our rank */
	int m_rank;

	/** #Processes */
	int m_numProcs;


	/** The number of vertices after filtering */
	unsigned int m_numFilterVertices;

	/** The global id for each vertex */
	unsigned long* m_globalIds;

	/** A list of local vertex ids that are duplicates */
	unsigned int* m_duplicates;

public:
	ParallelVertexFilter()
		: m_rank(0),
		m_numProcs(1),
		m_numFilterVertices(0),
		m_globalIds(0L),
		m_duplicates(0L)
	{
#ifdef USE_MPI
		setComm(MPI_COMM_WORLD);
#endif // USE_MPI
	}

	virtual ~ParallelVertexFilter()
	{
		delete [] m_globalIds;
		delete [] m_duplicates;
	}

#ifdef USE_MPI
	/**
	 * Call this before calling {@link filter} to use a different communicator than
	 * MPI_COMM_WORLD.
	 */
	void setComm(MPI_Comm comm)
	{
		m_comm = comm;
		MPI_Comm_rank(comm, &m_rank);
		MPI_Comm_size(comm, &m_numProcs);
	}
#endif // USE_MPI

	/**
	 * @param vertices Vertices that should be filtered, must have the size <code>numVertices * 3</code>
	 */
	void filter(unsigned int numVertices, const T *vertices)
	{
		SCOREP_USER_REGION("ParallelVertexFilter_Filter", SCOREP_USER_REGION_TYPE_FUNCTION);

#ifdef USE_MPI
		// MPI data type consisting of three doubles/floats
		MPI_Datatype vertexType;
		MPI_Type_contiguous(3, mpiFloatType(), &vertexType);
		MPI_Type_commit(&vertexType);

		// Chop the last 4 bits to avoid numerical errors
		T *roundVertices = new T[numVertices*3];
		removeRoundError(vertices, numVertices*3, roundVertices);

		// Create indices and sort them locally
		unsigned int *sortIndices = new unsigned int[numVertices];
		IndexSort<UNSTABLE, T>::sort(roundVertices, numVertices, sortIndices);

		// Select BUCKETS_PER_RANK-1 splitter elements
		T localSplitters[BUCKETS_PER_RANK-1];
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
		T *allSplitters = 0L;

		if (m_rank == 0)
			allSplitters = new T[m_numProcs * (BUCKETS_PER_RANK-1)];

		MPI_Gather(localSplitters, BUCKETS_PER_RANK-1, mpiFloatType(),
				allSplitters, BUCKETS_PER_RANK-1, mpiFloatType(),
				0, m_comm);

		// Sort splitter elements
		if (m_rank == 0)
			std::sort(allSplitters, allSplitters + (m_numProcs * (BUCKETS_PER_RANK-1)));

		// Distribute splitter to all processes
		T *splitters = new T[m_numProcs-1];

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

		MPI_Bcast(splitters, m_numProcs-1, mpiFloatType(), 0, m_comm);

		delete [] allSplitters;

		// Determine the bucket for each vertex
		unsigned int *bucket = new unsigned int[numVertices];

#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++) {
			T* ub = std::upper_bound(splitters, splitters+m_numProcs-1, roundVertices[i*3]);

			bucket[i] = ub-splitters;
		}

		delete [] splitters;

		// Determine the (local and total) bucket size
		int *bucketSize = new int[m_numProcs];
		memset(bucketSize, 0, sizeof(int)*m_numProcs);
		for (unsigned int i = 0; i < numVertices; i++) {
			assert(bucket[i] < static_cast<unsigned int>(m_numProcs));
			bucketSize[bucket[i]]++;
		}

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
		T *sendVertices = new T[3 * numVertices];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++) {
			memcpy(&sendVertices[i*3], &roundVertices[sortIndices[i]*3], sizeof(T)*3);
		}

		delete [] roundVertices;

		// Allocate buffer for the vertices and exchange them
		T *sortVertices = new T[3 * numSortVertices];

		int *sDispls = new int[m_numProcs];
		int *rDispls = new int[m_numProcs];
		sDispls[0] = 0;
		rDispls[0] = 0;
		for (int i = 1; i < m_numProcs; i++) {
			sDispls[i] = sDispls[i-1] + bucketSize[i-1];
			rDispls[i] = rDispls[i-1] + recvSize[i-1];
		}
		MPI_Alltoallv(sendVertices, bucketSize, sDispls, vertexType,
				sortVertices, recvSize, rDispls, vertexType, m_comm);

		delete [] sendVertices;

		// Create indices and sort them (such that the vertices are sorted)
		unsigned int *sortSortIndices = new unsigned int[numSortVertices];
		IndexSort<STABLE, T>::sort(sortVertices, numSortVertices, sortSortIndices);

		// A list that indicates which vertex is a duplicate (with use char instead of bool to work with MPI)
		char* sortDuplicate = new char[numSortVertices];

		if (numSortVertices > 0) {
			sortDuplicate[sortSortIndices[0]] = 0;
			for (unsigned int i = 1; i < numSortVertices; i++) {
				if (equals(&sortVertices[sortSortIndices[i-1]*3], &sortVertices[sortSortIndices[i]*3]))
					sortDuplicate[sortSortIndices[i]] = 1;
				else
					sortDuplicate[sortSortIndices[i]] = 0;
			}
		}

		delete [] sortVertices;

		// Send back the duplicate information
		char* duplicate = new char[numVertices];
		MPI_Alltoallv(sortDuplicate, recvSize, rDispls, MPI_CHAR,
				duplicate, bucketSize, sDispls, MPI_CHAR, m_comm);

		// Count the number of duplicates
		m_numFilterVertices = std::count_if(duplicate, duplicate+numVertices, isZero);
		assert(m_numFilterVertices <= numVertices);

		// Get the vertices offset
		unsigned long offset = m_numFilterVertices;
		MPI_Scan(MPI_IN_PLACE, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
		offset -= m_numFilterVertices;

		// Create the inverse sortIndex
		unsigned int* invSortIndices = new unsigned int[numVertices];
		for (unsigned int i = 0; i < numVertices; i++) {
			assert(sortIndices[i] < numVertices);
			invSortIndices[sortIndices[i]] = i;
		}

		// Set the global ids for all non duplicates and create the list of duplicates
		delete [] m_globalIds;
		m_globalIds = new unsigned long[numVertices];
		delete [] m_duplicates;
		m_duplicates = new unsigned int[numVertices - m_numFilterVertices];

		unsigned int dupPos = 0;
		for (unsigned int i = 0; i < numVertices; i++) {
			assert(invSortIndices[i] < numVertices);
			if (duplicate[invSortIndices[i]]) {
				m_globalIds[i] = std::numeric_limits<unsigned long>::max();
				assert(dupPos < numVertices - m_numFilterVertices);
				m_duplicates[dupPos++] = i;
			} else {
				m_globalIds[i] = offset++;
			}
		}

		delete [] invSortIndices;

		// Sent the global ids to the sorting ranks
		unsigned long* sendrecvGlobalIds = new unsigned long[numVertices];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++) {
			assert(sortIndices[i] < numVertices);
			sendrecvGlobalIds[i] = m_globalIds[sortIndices[i]];
		}

		unsigned long* sortGlobalIds = new unsigned long[numSortVertices];
		MPI_Alltoallv(sendrecvGlobalIds, bucketSize, sDispls, MPI_UNSIGNED_LONG,
				sortGlobalIds, recvSize, rDispls, MPI_UNSIGNED_LONG, m_comm);

		// Fill missing global ids
		for (unsigned int i = 1; i < numSortVertices; i++) {
			assert(sortSortIndices[i] < numSortVertices);
			if (sortDuplicate[sortSortIndices[i]]) {
				assert(sortGlobalIds[sortSortIndices[i]] == std::numeric_limits<unsigned long>::max());
				sortGlobalIds[sortSortIndices[i]] = sortGlobalIds[sortSortIndices[i-1]];
			}
		}

		delete [] sortDuplicate;
		delete [] sortSortIndices;

		// Send back the global ids (including ids for duplicate verices)
		MPI_Alltoallv(sortGlobalIds, recvSize, rDispls, MPI_UNSIGNED_LONG,
				sendrecvGlobalIds, bucketSize, sDispls, MPI_UNSIGNED_LONG, m_comm);

		delete [] sortGlobalIds;

#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++) {
			assert(sortIndices[i] < numVertices);
			assert(m_globalIds[sortIndices[i]] == std::numeric_limits<unsigned long>::max()
				|| m_globalIds[sortIndices[i]] == sendrecvGlobalIds[i]);
			m_globalIds[sortIndices[i]] = sendrecvGlobalIds[i];
		}

		delete [] sendrecvGlobalIds;

		delete [] bucketSize;
		delete [] recvSize;
		delete [] sDispls;
		delete [] rDispls;

		delete [] sortIndices;

		MPI_Type_free(&vertexType);
#endif // USE_MPI
	}

	/**
	 * @return Number of vertices this process is responsible for after filtering
	 */
	unsigned int numLocalVertices() const
	{
		return m_numFilterVertices;
	}

	/**
	 * @return The list of the global identifiers after filtering
	 */
	const unsigned long* globalIds() const
	{
		return m_globalIds;
	}

	/**
	 * @return The list of local vertex ids that are duplicates. The list has length
	 *  <code>numVertices</code> - {@link numLocalVertices}.
	 */
	const unsigned int* duplicates() const
	{
		return m_duplicates;
	}

private:
#ifdef USE_MPI
	/**
	 * @return The MPI datatype for <code>T</code>
	 */
	static MPI_Datatype mpiFloatType();
#endif // USE_MPI

	/**
	 * Removes round errors of double values by setting the last 4 bits
	 * (of the significand) to zero.
	 *
	 * @warning Only works if <code>value</code> ist not nan or infinity
	 * @todo This should work for arbitrary precision
	 */
	static T removeRoundError(T value)
	{
		FloatUnion<T> result;
		result.f = value;

		result.bits &= ~0xFF;

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
	static void removeRoundError(const T* values, unsigned int count, T* roundValues)
	{
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < count; i++)
			roundValues[i] = removeRoundError(values[i]);
	}

	/**
	 * Compares to vertices for equality
	 * Assumes that the rounding errors are removed.
	 */
	static bool equals(const T* vertexA, const T* vertexB)
	{
		return vertexA[0] == vertexB[0]
		       && vertexA[1] == vertexB[1]
		       && vertexA[2] == vertexB[2];
	}

	static bool isZero(char c)
	{
		return c == 0;
	}

	/** The total buckets we create is <code>BUCKETS_PER_RANK * numProcs</code> */
	const static int BUCKETS_PER_RANK = 8;
};

#ifdef USE_MPI
template<> inline
MPI_Datatype ParallelVertexFilter<float>::mpiFloatType()
{
	return MPI_FLOAT;
}

template<> inline
MPI_Datatype ParallelVertexFilter<double>::mpiFloatType()
{
	return MPI_DOUBLE;
}
#endif // USE_MPI

}

}

#endif // XDMFWRITER_PARALLELVERTEXFILTER_H
