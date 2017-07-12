/**
 * @file
 *  This file is part of XdmfWriter
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 *
 * @copyright Copyright (c) 2016-2017, Technische Universitaet Muenchen.
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

#ifndef XDMF_WRITER_BACKENDS_BASE_H
#define XDMF_WRITER_BACKENDS_BASE_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <string>
#include <vector>
#include <sys/stat.h>

#include "utils/env.h"
#include "utils/logger.h"
#include "utils/mathutils.h"
#include "utils/timeutils.h"

#ifdef USE_MPI
#include "BlockBuffer.h"
#else // USE_MPI
#include "BlockBufferSerial.h"
#endif // USE_MPI

namespace xdmfwriter
{

namespace backends
{

enum VariableType
{
	FLOAT, // float or double
	INT,
	UNSIGNED_LONG
};

struct VariableData
{
	VariableData(const char* name, VariableType type, unsigned int count, bool hasTime)
		: name(name), type(type), count(count), hasTime(hasTime)
	{ }

	const char* name;

	VariableType type;

	/** The number of values in the variable */
	unsigned int count;

	bool hasTime;
};

template<typename T>
class Base
{
private:
	const std::string m_format;

#ifdef USE_MPI
	MPI_Comm m_comm;

	/** Communicator containing only I/O processes */
	MPI_Comm m_ioComm;
#endif // USE_MPI

	/** Rank of this process */
	int m_rank;

	std::string m_pathPrefix;

	/** The file name prefix but without directories (required for the XML file) */
	std::string m_filePrefix;

	std::vector<VariableData> m_variableData;

	/** The file system block size */
	unsigned long m_blockSize;

	/** Total number of cells */
	unsigned long m_totalElems;
	/** Local number of cells */
	unsigned int m_localElems;
	/** Offset where we start writing our part */
	unsigned long m_offset;

	/** Block buffer used to create equal sized blocks */
	BlockBuffer m_blockBuffer;

	/** The buffer size required for the block buffer */
	unsigned long m_bufferSize;

protected:
	Base(const char* format)
		: m_format(format),
#ifdef USE_MPI
		  m_comm(MPI_COMM_WORLD),
		  m_ioComm(MPI_COMM_NULL),
#endif // USE_MPI
		  m_rank(0),
		  m_blockSize(1),
		  m_totalElems(0),
		  m_localElems(0),
		  m_offset(0),
		  m_bufferSize(0)
	{
	}

public:
	virtual ~Base()
	{
		close();
	}

#ifdef USE_MPI
	/**
	 * Sets the communicator that should be used. Default is MPI_COMM_WORLD.
	 */
	void setComm(MPI_Comm comm)
	{
		m_comm = comm;
		MPI_Comm_rank(comm, &m_rank);
	}
#endif // USE_MPI

	virtual void open(const std::string &outputPrefix, const std::vector<VariableData> &variableData, bool create = true)
	{
		// Store file and backend prefix
		m_pathPrefix = outputPrefix;
		m_filePrefix = outputPrefix;
		// Remove all directories from prefix
		size_t pos = m_filePrefix.find_last_of('/');
		if (pos != std::string::npos)
			m_filePrefix = m_filePrefix.substr(pos+1);

		m_variableData = variableData;

		m_blockSize = utils::Env::get<unsigned long>("XDMFWRITER_BLOCK_SIZE", 1);
	}

	virtual void setMesh(unsigned int meshId,
		unsigned long totalElements, unsigned int localElements, unsigned long offset) = 0;

	/**
	 * @param buffer A buffer that is large if enough and can be used for temporary storage
	 */
	void writeData(unsigned int timestep, unsigned int id, const void* data, void* buffer)
	{
#ifdef USE_MPI
		if (m_blockBuffer.isInitialized()) {
			m_blockBuffer.exchange(data, variableMPIType(m_variableData[id].type),
				m_variableData[id].count, buffer);
			write(timestep, id, buffer);
		} else {
#endif // USE_MPI
			write(timestep, id, data);
#ifdef USE_MPI
		}
#endif // USE_MPI
	}

	virtual void flush() = 0;

	virtual void close()
	{
#ifdef USE_MPI
		if (m_ioComm != MPI_COMM_NULL && m_ioComm != m_comm) {
			MPI_Comm_free(&m_ioComm);
			m_ioComm = MPI_COMM_NULL;
		}
#endif // USE_MPI
	}

	unsigned long bufferSize() const
	{
		return m_bufferSize;
	}

	const char* format() const
	{
		return m_format.c_str();
	}

	unsigned int numVariables() const
	{
		return m_variableData.size();
	}

	/**
	 * @return Total number of elements after alignment (including alignment)
	 */
	unsigned long totalElements() const
	{
		return m_totalElems;
	}

	/**
	 * @return The relative (to the XDMF file) data location in XMDF format
	 *
	 * @warning Subclasses have to override this function to return the complete location
	 */
	virtual std::string dataLocation(unsigned int meshId, const char* variable) const
	{
		return m_filePrefix;
	}

protected:
	void setMesh(unsigned long totalElements, unsigned int localElements, unsigned long offset, bool alignTotalElements = false)
	{
		m_totalElems = totalElements;
		m_localElems = localElements;
		m_offset = offset;

		m_bufferSize = 0;
#ifdef USE_MPI
		// Create block buffer
		if (m_blockSize > 1) {
			m_blockBuffer.init(m_comm, localElements, sizeof(T), m_blockSize);

			if (m_ioComm != MPI_COMM_NULL)
				MPI_Comm_free(&m_ioComm);

			MPI_Comm_split(m_comm, m_blockBuffer.count() > 0 ? 1 : MPI_UNDEFINED, 0, &m_ioComm);
			m_localElems = m_blockBuffer.count();

			if (alignTotalElements) {
				m_totalElems = utils::MathUtils::roundUp(m_totalElems, Base<T>::blockSize() / sizeof(T));
				// This will only change on the last rank
				m_localElems = utils::MathUtils::roundUp(m_localElems, static_cast<unsigned int>(Base<T>::blockSize() / sizeof(T)));
			}

			m_offset = m_localElems;
			if (m_offset > 0) {
				MPI_Scan(MPI_IN_PLACE, &m_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, m_ioComm);
				m_offset -= m_localElems;
			}

			// Compute the max buffer space we need
			for (std::vector<VariableData>::const_iterator it = m_variableData.begin();
					it != m_variableData.end(); ++it) {
				m_bufferSize = std::max(m_bufferSize,
					static_cast<unsigned long>(it->count * variableSize(it->type)));
			}
			m_bufferSize *= m_localElems;
		} else {
			m_ioComm = m_comm;
		}
#endif // USE_MPI
	}

	/**
	 * Backup an existing backend file
	 */
	void backup(const std::string &file)
	{
		// Backup any existing file
		struct stat statBuffer;
		if (m_rank == 0 && stat(file.c_str(), &statBuffer) == 0) {
			logWarning() << file << "already exists. Creating backup.";
			rename(file.c_str(), (file + ".bak_" + utils::TimeUtils::timeAsString("%F_%T", time(0L))).c_str());
		}
	}

#ifdef USE_MPI
	MPI_Comm comm() const
	{
		return m_comm;
	}

	/**
	 * The communicator that only includes
	 * the I/O ranks
	 */
	MPI_Comm ioComm() const
	{
		return m_ioComm;
	}
#endif // USE_MPI

	int rank() const
	{
		return m_rank;
	}

	const std::string& pathPrefix() const
	{
		return m_pathPrefix;
	}

	/**
	 * @return The local number of elements (including alignment)
	 */
	unsigned int localElements() const
	{
		return m_localElems;
	}

	unsigned long offset() const
	{
		return m_offset;
	}

	unsigned long blockSize() const
	{
		return m_blockSize;
	}

	const std::vector<VariableData>& variables() const
	{
		return m_variableData;
	}

	virtual void write(unsigned int timestep, unsigned int id, const void* data) = 0;

protected:
	static unsigned int variableSize(VariableType type)
	{
		switch (type) {
		case FLOAT:
			return sizeof(T);
		case INT:
			return sizeof(int);
		case UNSIGNED_LONG:
			return sizeof(unsigned long);
		}

		return 0;
	}

#ifdef USE_MPI
	static MPI_Datatype variableMPIType(VariableType type)
	{
		switch (type) {
		case FLOAT:
			if (sizeof(T) == sizeof(float))
				return MPI_FLOAT;
			return MPI_DOUBLE;
		case INT:
			return MPI_INT;
		case UNSIGNED_LONG:
			return MPI_UNSIGNED_LONG;
		}

		return MPI_DATATYPE_NULL;
	}
#endif // USE_MPI
};

}

}

#endif // XDMF_WRITER_BACKENDS_BASE_H
