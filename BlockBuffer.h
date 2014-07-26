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

#ifndef BLOCK_BUFFER_H
#define BLOCK_BUFFER_H

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdint.h>

#include "utils/logger.h"

/**
 * Splits a buffer into equal sized blocks and distributes the
 * blocks across all ranks
 */
class BlockBuffer
{
private:
	/** Our rank */
	int m_rank;

	/** The basic data type */
	MPI_Datatype m_type;

	/** Size of the data type */
	MPI_Aint m_extent;

	/** Number of elements before the exchange */
	unsigned int m_inCount;

	/** Rank to which we send our data */
	int m_sendRank;

	/** Number of elements we send to the left */
	unsigned int m_sendCount;

	/** Number of ranks from which we receive */
	unsigned int m_recvRanks;

	/** Number of elements we receive from the right */
	unsigned int *m_recvCounts;

	/** Number of elements we will have after and exchange */
	unsigned int m_outCount;

	/** Requests used for sending/receiving (all allocate them once) */
	MPI_Request *m_requests;

public:
	BlockBuffer()
		: m_rank(0), m_type(MPI_DATATYPE_NULL), m_extent(0), m_inCount(0), m_sendRank(0),
		  m_sendCount(0), m_recvRanks(0), m_recvCounts(0L), m_outCount(0), m_requests(0L)
	{
	}

	virtual ~BlockBuffer()
	{
		delete [] m_recvCounts;
	}

	/**
	 * @param nLocalElements The number of elements in the local buffer
	 * @param dataType The MPI data type of the elements
	 * @param blockSize The size of the blocks that should be created (in bytes)
	 */
	void init(unsigned int inCount, MPI_Datatype dataType, unsigned long blockSize)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

		MPI_Aint lb;
		MPI_Type_get_extent(dataType, &lb, &m_extent);
		m_type = dataType;

		if (blockSize % m_extent != 0)
			logError() << "The block size must be a multiple of the data type size";

		unsigned long localSize = inCount * m_extent;
		unsigned long localStart;
		MPI_Scan(&localSize, &localStart, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
		localStart -= localSize;

		// Determine what we need to send, what is ours and what we receive
		unsigned long sendSize = (blockSize - localStart % blockSize) % blockSize;
		unsigned long recvSize = (blockSize - (localStart + localSize) % blockSize) % blockSize;
		assert((localStart - sendSize + recvSize) % blockSize == 0);
		unsigned int localBlocks = (localSize - sendSize + recvSize) / blockSize;

		m_sendCount = sendSize / m_extent;
		m_sendCount = std::min(m_sendCount, inCount);

		// Compute all the ranks, from which we receive and to which we send
		int numProcs;
		MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

		unsigned int *blocks = new unsigned int[numProcs];
		MPI_Allgather(&localBlocks, 1, MPI_UNSIGNED, blocks, 1, MPI_UNSIGNED, MPI_COMM_WORLD);

		m_sendRank = m_rank-1;
		if (m_rank > 0) {
			// Only if we have a left process
			while (blocks[m_sendRank] == 0) {
				m_sendRank--;
				assert(m_sendRank >= 0);
			}
		}

		int lastRecvRank = m_rank+1;
		if (localBlocks == 0 || m_rank == numProcs-1)
			lastRecvRank = m_rank;
		else {
			// Only if we receive any elements
			while (blocks[lastRecvRank] == 0 && lastRecvRank+1 < numProcs)
				lastRecvRank++;
		}
		m_recvRanks = lastRecvRank - m_rank;

		delete [] blocks;

		// Exchange the elements we transfer from/to each rank
		m_requests = new MPI_Request[m_recvRanks + 1];
		m_recvCounts = new unsigned int[m_recvRanks];
		if (m_rank == 0)
			m_requests[0] = MPI_REQUEST_NULL;
		else
			MPI_Isend(&m_sendCount, 1, MPI_UNSIGNED, m_sendRank, 0,
					MPI_COMM_WORLD, &m_requests[0]);
		for (unsigned int i = 0; i < m_recvRanks; i++)
			MPI_Irecv(&m_recvCounts[i], 1, MPI_UNSIGNED, m_rank+i+1, 0,
					MPI_COMM_WORLD, &m_requests[i+1]);
		MPI_Waitall(m_recvRanks+1, m_requests, MPI_STATUSES_IGNORE); // Important: This sets all requests to MPI_REQEUST_NULL

		// Compute how many elements we are responsible for
		m_inCount = inCount;
		m_outCount = inCount - m_sendCount;
		for (unsigned int i = 0; i < m_recvRanks; i++)
			m_outCount += m_recvCounts[i];
	}

	/**
	 * @return The number of elements after the exchange
	 */
	unsigned int count() const
	{
		return m_outCount;
	}

	/**
	 * Exchanges the data according to the configuration
	 *
	 * @in The local data before the exchange.
	 * @out The local data after the exchange. The caller is responsible
	 *  for allocating this buffer.
	 */
	void exchange(void* in, void* out)
	{
		// Recv elements
		unsigned int elemOffset = m_inCount-m_sendCount;
		for (unsigned int i = 0; i < m_recvRanks; i++) {
			if (m_recvCounts[i] > 0) {
				MPI_Irecv(static_cast<uint8_t*>(out)+elemOffset*m_extent,
						m_recvCounts[i], m_type, m_rank+i+1, 0, MPI_COMM_WORLD, &m_requests[i+1]);
				elemOffset += m_recvCounts[i];
			}
		}

		// Send first elements
		if (m_sendCount > 0)
			MPI_Isend(in, m_sendCount, m_type, m_sendRank, 0,
					MPI_COMM_WORLD, &m_requests[0]);

		// Copy local elements
		memcpy(out, static_cast<uint8_t*>(in)+m_sendCount*m_extent,
				(m_inCount-m_sendCount)*m_extent);

		MPI_Waitall(m_recvRanks+1, m_requests, MPI_STATUSES_IGNORE);
	}
};

#endif // BLOCK_BUFFER_H
