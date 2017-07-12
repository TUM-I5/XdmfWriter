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

#ifndef XDMF_WRITER_BACKENDS_BACKEND_H
#define XDMF_WRITER_BACKENDS_BACKEND_H

#ifdef USE_HDF
#include "HDF5.h"
#endif // USE_HDF
#include "Posix.h"

namespace xdmfwriter
{

/**
 * The backend tpyes
 */
enum BackendType {
#ifdef USE_HDF
	H5,
#endif // USE_HDF
	POSIX
};

namespace backends
{

template<typename T>
class Backend
{
private:
	/** Storage for cell data */
	Base<T>* const m_cellData;

	/** Storage for vertex data */
	Base<T>* const m_vertexData;

	/** The buffer for the block buffer */
	void* m_buffer;

public:
	Backend(BackendType backendType)
		: m_cellData(createHeavyDataStorage(backendType)),
		m_vertexData(createHeavyDataStorage(backendType)),
		m_buffer(0L)
	{
	}

	virtual ~Backend()
	{
		delete m_cellData;
		delete m_vertexData;

		free(m_buffer);
	}

#ifdef USE_MPI
	void setComm(MPI_Comm comm)
	{
		m_cellData->setComm(comm);
		m_vertexData->setComm(comm);
	}
#endif // USE_MPI

	void open(const std::string &outputPrefix,
			const std::vector<VariableData> &cellVariableData, const std::vector<VariableData> &vertexVariableData,
			bool create = true)
	{
		m_cellData->open(outputPrefix + "_cell", cellVariableData, create);
		m_vertexData->open(outputPrefix + "_vertex", vertexVariableData, create);
	}

	/**
	 * @param meshId The id of the new mesh
	 */
	void setMesh(unsigned int meshId,
		const unsigned long totalSize[2], const unsigned int localSize[2],
		const unsigned long offset[2])
	{
		m_cellData->setMesh(meshId, totalSize[0], localSize[0], offset[0]);
		m_vertexData->setMesh(meshId, totalSize[1], localSize[1], offset[1]);

		m_buffer = realloc(m_buffer, std::max(m_cellData->bufferSize(), m_vertexData->bufferSize()));
	}

	void writeCellData(unsigned int timestep, unsigned int id, const void* data)
	{
		m_cellData->writeData(timestep, id, data, m_buffer);
	}

	void writeVertexData(unsigned int timestep, unsigned int id, const void* data)
	{
		m_vertexData->writeData(timestep, id, data, m_buffer);
	}

	void flush()
	{
		m_cellData->flush();
		m_vertexData->flush();
	}

	void close()
	{
		m_cellData->close();
		m_vertexData->close();
	}

	/**
	 * The name of the format in the XML file
	 */
	const char* format() const
	{
		return m_cellData->format();
	}

	/**
	 * @return The number of cell variables
	 */
	unsigned int numCellVars() const
	{
		return m_cellData->numVariables();
	}

	/**
	 * @return The number of vertex variables
	 */
	unsigned int numVertexVars() const
	{
		return m_vertexData->numVariables();
	}

	unsigned long numAlignedCells() const
	{
		return m_cellData->totalElements();
	}

	unsigned long numAlignedVertices() const
	{
		return m_vertexData->totalElements();
	}

	std::string cellDataLocation(unsigned int meshId, const char* variable) const
	{
		return m_cellData->dataLocation(meshId, variable);
	}

	std::string vertexDataLocation(unsigned int meshId, const char* variable) const
	{
		return m_vertexData->dataLocation(meshId, variable);
	}

private:
	static Base<T>* createHeavyDataStorage(BackendType type)
	{
		switch(type) {
		case POSIX:
			return new Posix<T>();
#ifdef USE_HDF
		case H5:
			return new HDF5<T>();
			break;
#endif // USE_HDF
		default:
			logError() << "Unknown backend type";
		}

		return 0L;
	}
};

}

}

#endif // XDMF_WRITER_BACKENDS_BACKEND_H
