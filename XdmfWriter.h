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

#ifndef XDMFWRITER_XDMFWRITER_H
#define XDMFWRITER_XDMFWRITER_H

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "utils/env.h"
#include "utils/logger.h"
#include "utils/mathutils.h"

#include "scorep_wrapper.h"
#include "BufferFilter.h"
#include "ParallelVertexFilter.h"
#include "Topology.h"
#include "backends/Backend.h"

namespace xdmfwriter
{

/**
 * Writes data in XDMF format
 */
template<TopoType Topo, typename T>
class XdmfWriter
{
private:
#ifdef USE_MPI
	MPI_Comm m_comm;
#endif // USE_MPI

	int m_rank;

	std::string m_outputPrefix;

	std::fstream m_xdmfFile;

	/** The backend for large scale I/O */
	backends::Backend<T> m_backend;

	/** Names of the cell variables that should be written */
	std::vector<const char*> m_cellVariableNames;

	/** Names of the vertex variables that should be written */
	std::vector<const char*> m_vertexVariableNames;

	/** Vertex filter (only used if vertex filter is enabled) */
	internal::ParallelVertexFilter<T> m_vertexFilter;

	/** The buffer filter for vertex data (only used if vertex filter is enabled) */
	internal::BufferFilter<sizeof(T)> m_vertexDataFilter;

	/** Only execute the flush on certain time steps */
	unsigned int m_flushInterval;

	/** Output step counter */
	unsigned int m_timeStep;

	/** The current mesh id */
	unsigned int m_meshId;

	/** The timestep counter of the current mesh */
	unsigned int m_meshTimeStep;

	bool m_useVertexFilter;

	bool m_writePartitionInfo;

	/** Total number of cells/vertices */
	unsigned long m_totalSize[2];

public:
	/**
	 * @param timestep Set this to > 0 to activate append mode
	 */
	XdmfWriter(BackendType backendType,
			const char* outputPrefix,
			unsigned int timeStep = 0)
		: m_rank(0), m_outputPrefix(outputPrefix),
		m_backend(backendType),
		m_flushInterval(0),
		m_timeStep(timeStep), m_meshId(0), m_meshTimeStep(0),
		m_useVertexFilter(true), m_writePartitionInfo(true)
	{
#ifdef USE_MPI
		setComm(MPI_COMM_WORLD);
#endif // USE_MPI
	}

	virtual ~XdmfWriter()
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

		m_backend.setComm(comm);
		m_vertexFilter.setComm(comm);
	}
#endif // USE_MPI

	void init(const std::vector<const char*> &cellVariableNames, const std::vector<const char*> &vertexVariableNames,
			bool useVertexFilter = true, bool writePartitionInfo = true)
	{
		m_cellVariableNames = cellVariableNames;
		m_vertexVariableNames = vertexVariableNames;

		m_useVertexFilter = useVertexFilter;
		m_writePartitionInfo = writePartitionInfo;

		int nProcs = 1;
#ifdef USE_MPI
		MPI_Comm_size(m_comm, &nProcs);
#endif // USE_MPI
		if (nProcs == 1)
			m_useVertexFilter = false;

		// Create variable data for the backend
		std::vector<backends::VariableData> cellVariableData;
		cellVariableData.push_back(backends::VariableData("connect", backends::UNSIGNED_LONG, internal::Topology<Topo>::size(), false));
		if (writePartitionInfo)
			cellVariableData.push_back(backends::VariableData("partition", backends::INT, 1, false));
		for (std::vector<const char*>::const_iterator it = cellVariableNames.begin();
				it != cellVariableNames.end(); ++it) {
			cellVariableData.push_back(backends::VariableData(*it, backends::FLOAT, 1, true));
		}

		std::vector<backends::VariableData> vertexVariableData;
		vertexVariableData.push_back(backends::VariableData("geometry", backends::FLOAT, 3, false));
		for (std::vector<const char*>::const_iterator it = vertexVariableNames.begin();
				it != vertexVariableNames.end(); ++it) {
			vertexVariableData.push_back(backends::VariableData(*it, backends::FLOAT, 1, true));
		}

		// Open the backend
		m_backend.open(m_outputPrefix, cellVariableData, vertexVariableData, m_timeStep == 0);

		// Write the XML file
		if (m_rank == 0) {
			std::string xdmfName = m_outputPrefix + ".xdmf";

			std::ofstream(xdmfName.c_str(), std::ios::app).close(); // Create the file (if it does not exist)
			m_xdmfFile.open(xdmfName.c_str());

			m_xdmfFile << "<?xml version=\"1.0\" ?>" << std::endl
					<< "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl
					<< "<Xdmf Version=\"2.0\">" << std::endl
					<< " <Domain>" << std::endl
					<< "  <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

			if (m_timeStep == 0)
				closeXdmf();
			else {
				// Jump the correct position in the file
				std::ostringstream tStartStream;
				timeStepStartXdmf(m_timeStep-1, tStartStream);
				std::string tStart = tStartStream.str();

				// Find beginning of the (correct) time step
				std::string line;
				std::size_t pos;
				while (getline(m_xdmfFile, line)) {
					pos = line.find(tStart);
					if (pos != std::string::npos)
						break;
				}
				if (!m_xdmfFile)
					logError() << "Unable to find time step for appending";

				// Extract mesh id and mesh step
				std::istringstream ss(line.substr(pos + tStart.size()));
				ss.seekg(13, std::iostream::cur); // Skip "<!-- mesh id: "
				ss >> m_meshId;
				ss.seekg(13, std::iostream::cur); // Skip ", mesh step: "
				ss >> m_meshTimeStep;
				logInfo() << "Found mesh" << m_meshId << "in step" << m_meshTimeStep;

				m_meshId++;
				m_meshTimeStep++;

				// Find end of this time step
				while (getline(m_xdmfFile, line)) {
					if (line.find("</Grid>") != std::string::npos)
						break;
				}
			}
		}

#ifdef USE_MPI
		if (m_timeStep != 0) {
			// Broadcast the some information if we restart
			unsigned int buf[2] = {m_meshId, m_meshTimeStep};
			MPI_Bcast(buf, 2, MPI_UNSIGNED, 0, m_comm);
			m_meshId = buf[0];
			m_meshTimeStep = buf[1];
		}
#endif // USE_MPI

		// Get flush interval
		m_flushInterval = utils::Env::get<unsigned int>("XDMFWRITER_FLUSH_INTERVAL", 1);
	}

	/**
	 * @param restarting Set this to <code>true</code> if the codes restarts from a checkpoint
	 *  and the XDMF writer should continue with the old mesh
	 */
	void setMesh(unsigned int numCells, const unsigned int* cells, unsigned int numVertices, const T *vertices, bool restarting = false)
	{
#ifdef USE_MPI
		// Apply vertex filter
		internal::BufferFilter<3*sizeof(T)> vertexRemover;
		if (m_useVertexFilter) {
			// Filter duplicate vertices
			m_vertexFilter.filter(numVertices, vertices);

			if (!restarting) {
				vertexRemover.init(numVertices, numVertices - m_vertexFilter.numLocalVertices(), m_vertexFilter.duplicates());
				vertices = static_cast<const T*>(vertexRemover.filter(vertices));
			}

			// Set the vertex data filter
			if (m_backend.numVertexVars() > 0)
				m_vertexDataFilter.init(numVertices, numVertices - m_vertexFilter.numLocalVertices(), m_vertexFilter.duplicates());

			numVertices = m_vertexFilter.numLocalVertices();
		}
#endif // USE_MPI

		// Set the backend mesh
		m_totalSize[0] = numCells;
		m_totalSize[1] = numVertices;
		unsigned long offset[2] = {numCells, numVertices};
#ifdef USE_MPI
		MPI_Allreduce(MPI_IN_PLACE, m_totalSize, 2, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
		MPI_Scan(MPI_IN_PLACE, offset, 2, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
#endif // USE_MPI
		offset[0] -= numCells;
		offset[1] -= numVertices;

		// Add a new mesh to the backend
		unsigned int localSize[2] = {numCells, numVertices};
		// Use the old mesh id for restarts
		m_backend.setMesh((restarting ? m_meshId-1 : m_meshId), m_totalSize, localSize, offset);

		if (restarting)
			// Can skip writing the mesh if we are restarting
			return;

		// Add vertex offset to all cells and convert to unsigned long
		unsigned long *h5Cells = new unsigned long[numCells * internal::Topology<Topo>::size()];
#ifdef USE_MPI
		if (m_useVertexFilter) {
#ifdef _OPENMP
			#pragma omp parallel for schedule(static)
#endif // _OPENMP
			for (size_t i = 0; i < numCells*internal::Topology<Topo>::size(); i++)
				h5Cells[i] = m_vertexFilter.globalIds()[cells[i]];
		} else
#endif // USE_MPI
		{
#ifdef _OPENMP
			#pragma omp parallel for schedule(static)
#endif // _OPENMP
			for (size_t i = 0; i < numCells*internal::Topology<Topo>::size(); i++)
				h5Cells[i] = cells[i] + offset[1];
		}

		m_backend.writeCellData(0, 0, h5Cells);
		delete [] h5Cells;

		if (m_writePartitionInfo) {
			// Create partition information
			unsigned int *partInfo = new unsigned int[numCells];
#ifdef _OPENMP
			#pragma omp parallel for schedule(static)
#endif // _OPENMP
			for (unsigned int i = 0; i < numCells; i++)
				partInfo[i] = m_rank;

			m_backend.writeCellData(0, 1, partInfo);

			delete [] partInfo;
		}

		m_backend.writeVertexData(0, 0, vertices);

		m_meshId++;
		m_meshTimeStep = 0;
	}

	/**
	 * Add a new output time step
	 */
	void addTimeStep(double time)
	{
		if (m_rank == 0) {
			unsigned long alignedSize[2] = {m_backend.numAlignedCells(), m_backend.numAlignedVertices()};

			m_xdmfFile << "   ";
			timeStepStartXdmf(m_timeStep, m_xdmfFile);
			// Generate information for restarting (WARNING: if this line is modified the initialization has to be adapted)
			m_xdmfFile << "<!-- mesh id: " << (m_meshId-1) << ", mesh step: " << m_meshTimeStep << " -->";
			m_xdmfFile << std::endl;
			m_xdmfFile << "    <Topology TopologyType=\"" << internal::Topology<Topo>::name() << "\" NumberOfElements=\"" << m_totalSize[0] << "\">" << std::endl
					// This should be UInt but for some reason this does not work with binary data
					<< "     <DataItem NumberType=\"Int\" Precision=\"8\" Format=\""
						<< m_backend.format() << "\" Dimensions=\"" << m_totalSize[0] << " " << internal::Topology<Topo>::size() << "\">"
						<< m_backend.cellDataLocation(m_meshId-1, "connect")
						<< "</DataItem>" << std::endl
					<< "    </Topology>" << std::endl
					<< "    <Geometry name=\"geo\" GeometryType=\"XYZ\" NumberOfElements=\"" << m_totalSize[1] << "\">" << std::endl
					<< "     <DataItem NumberType=\"Float\" Precision=\"" << sizeof(T) << "\" Format=\""
						<< m_backend.format() << "\" Dimensions=\"" << m_totalSize[1] << " 3\">"
						<< m_backend.vertexDataLocation(m_meshId-1, "geometry")
						<< "</DataItem>" << std::endl
					<< "    </Geometry>" << std::endl
					<< "    <Time Value=\"" << time << "\"/>" << std::endl;
			if (m_writePartitionInfo) {
				m_xdmfFile << "    <Attribute Name=\"partition\" Center=\"Cell\">" << std::endl
						<< "     <DataItem  NumberType=\"Int\" Precision=\"4\" Format=\""
							<< m_backend.format() << "\" Dimensions=\"" << m_totalSize[0] << "\">"
							<< m_backend.cellDataLocation(m_meshId-1, "partition")
							<< "</DataItem>" << std::endl
						<< "    </Attribute>" << std::endl;
			}
			for (size_t i = 0; i < m_cellVariableNames.size(); i++) {
				m_xdmfFile << "    <Attribute Name=\"" << m_cellVariableNames[i] << "\" Center=\"Cell\">" << std::endl
						<< "     <DataItem ItemType=\"HyperSlab\" Dimensions=\"" << m_totalSize[0] << "\">" << std::endl
						<< "      <DataItem NumberType=\"UInt\" Precision=\"4\" Format=\"XML\" Dimensions=\"3 2\">"
						<< m_meshTimeStep << " 0 1 1 1 " << m_totalSize[0] << "</DataItem>" << std::endl
						<< "      <DataItem NumberType=\"Float\" Precision=\"" << sizeof(T) << "\" Format=\""
							<< m_backend.format() << "\" Dimensions=\""
							<< (m_meshTimeStep + 1) << ' ' << alignedSize[0] << "\">"
							<< m_backend.cellDataLocation(m_meshId-1, m_cellVariableNames[i])
							<< "</DataItem>" << std::endl
						<< "     </DataItem>" << std::endl
						<< "    </Attribute>" << std::endl;
			}
			for (size_t i = 0; i < m_vertexVariableNames.size(); i++) {
				m_xdmfFile << "    <Attribute Name=\"" << m_vertexVariableNames[i] << "\" Center=\"Node\">" << std::endl
						<< "     <DataItem ItemType=\"HyperSlab\" Dimensions=\"" << m_totalSize[1] << "\">" << std::endl
						<< "      <DataItem NumberType=\"UInt\" Precision=\"4\" Format=\"XML\" Dimensions=\"3 2\">"
						<< m_meshTimeStep << " 0 1 1 1 " << m_totalSize[1] << "</DataItem>" << std::endl
						<< "      <DataItem NumberType=\"Float\" Precision=\"" << sizeof(T) << "\" Format=\""
							<< m_backend.format() << "\" Dimensions=\""
							<< (m_meshTimeStep + 1) << ' ' << alignedSize[1] << "\">"
							<< m_backend.vertexDataLocation(m_meshId-1, m_vertexVariableNames[i])
							<< "</DataItem>" << std::endl
						<< "     </DataItem>" << std::endl
						<< "    </Attribute>" << std::endl;
			}

			m_xdmfFile << "   </Grid>" << std::endl;

			closeXdmf();
		}

		m_timeStep++;
		m_meshTimeStep++;
	}

	/**
	 * Write cell data for one variable at the current time step
	 *
	 * @param id The number of the variable that should be written
	 */
	void writeCellData(unsigned int id, const T *data)
	{
		SCOREP_USER_REGION("XDMFWriter_writeCellData", SCOREP_USER_REGION_TYPE_FUNCTION);

		m_backend.writeCellData(m_meshTimeStep-1, id + (m_writePartitionInfo ? 2 : 1), data);
	}

	/**
	 * Write vertex data for one variable at the current time step
	 *
	 * @param id The number of the variable that should be written
	 */
	void writeVertexData(unsigned int id, const T *data)
	{
		SCOREP_USER_REGION("XDMFWriter_writeCellData", SCOREP_USER_REGION_TYPE_FUNCTION);

		// Filter duplicates if the vertex filter is enabled
		const void* tmp = data;
		if (m_useVertexFilter)
			tmp = m_vertexDataFilter.filter(data);

		m_backend.writeVertexData(m_meshTimeStep-1, id + 1, tmp);
	}

	/**
	 * Flushes the data to disk
	 */
	void flush()
	{
		SCOREP_USER_REGION("XDMFWriter_flush", SCOREP_USER_REGION_TYPE_FUNCTION);

		if (m_timeStep % m_flushInterval == 0)
			m_backend.flush();
	}

	/**
	 * Closes the HDF5 file (should be done before MPI_Finalize is called)
	 */
	void close()
	{
		// Close backend
		m_backend.close();
	}

	/**
	 * @return The current time step of the output file
	 */
	unsigned int timestep() const
	{
		return m_timeStep;
	}

private:
	void closeXdmf()
	{
		size_t contPos = m_xdmfFile.tellp();
		m_xdmfFile << "  </Grid>" << std::endl
				<< " </Domain>" << std::endl
				<< "</Xdmf>" << std::endl;
		m_xdmfFile.seekp(contPos);
	}

private:
	/**
	 * Write the beginning of a time step to the stream
	 */
	static void timeStepStartXdmf(unsigned int timestep, std::ostream &s)
	{
		s << "<Grid Name=\"step_" << std::setw(MAX_TIMESTEP_SPACE) << std::setfill('0') << timestep << std::setfill(' ')
				<< "\" GridType=\"Uniform\">";
	}

private:
	static const unsigned int MAX_TIMESTEP_SPACE = 12;
};

}

#endif // XDMFWRITER_XDMFWRITER_H
