/**
 * @file
 *  This file is part of XdmfWriter
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 *
 * @copyright Copyright (c) 2014-2015, Technische Universitaet Muenchen.
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

#ifndef XDMF_WRITER_H
#define XDMF_WRITER_H

#ifdef PARALLEL
#include <mpi.h>
#endif // PARALLEL

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#ifdef USE_HDF
#include <hdf5.h>
#endif // USE_HDF

#include "utils/env.h"
#include "utils/logger.h"

#include "epik_wrapper.h"
#include "scorep_wrapper.h"
#ifdef PARALLEL
#include "BlockBuffer.h"
#include "ParallelVertexFilter.h"
#else // PARALLEL
#include "BlockBufferSerial.h"
#endif // PARALLEL

namespace xdmfwriter
{

/**
 * The topology types
 */
enum TopoType {
	TRIANGLE,
	TETRAHEDRON
};

/**
 * Writes data in XDMF format
 */
template<enum TopoType>
class XdmfWriter
{
public:

private:
#ifdef PARALLEL
	MPI_Comm m_comm;
#endif // PARALLEL

	int m_rank;

	std::string m_outputPrefix;

	std::fstream m_xdmfFile;

#ifdef USE_HDF
	hid_t m_hdfFile;

	hid_t m_hdfAccessList;

	/** Variable identifiers */
	hid_t *m_hdfVars;
	/** Memory space for writing one time step */
	hid_t m_hdfVarMemSpace;
#endif // USE_HDF

	/** Name of the HDF5 file, without any directories (required in the XDMF file) */
	std::string m_hdfFilename;

	/** Names of the variables that should be written */
	const std::vector<const char*> m_variableNames;

	/** Total number of cells */
	unsigned long m_totalCells;
	/** Local number of cells */
	unsigned int m_localCells;
	/** Offset where we start writing our part */
	unsigned long m_offsetCells;

	/** Block buffer used to create equal sized blocks */
	BlockBuffer m_blockBuffer;

	/** Buffer required for blocking */
	double *m_blocks;

	/** Offsets in the XDMF file which describe the time dimension size */
	size_t *m_timeDimPos;

	/** Only execute the flush on certain time steps */
	unsigned int m_flushInterval;

	/** Output step counter */
	unsigned int m_timestep;

public:
	/**
	 * @param timestep Set this to > 0 to activate append mode
	 */
	XdmfWriter(int rank, const char* outputPrefix, const std::vector<const char*> &variableNames,
			unsigned int timestep = 0)
		: m_rank(rank), m_outputPrefix(outputPrefix), m_variableNames(variableNames),
		  m_totalCells(0), m_localCells(0), m_offsetCells(0),
		  m_blocks(0L), m_timeDimPos(0L), m_flushInterval(0), m_timestep(timestep)
	{
#ifdef PARALLEL
		m_comm = MPI_COMM_WORLD;
#endif // PARALLEL

#ifdef USE_HDF
		std::string prefix(outputPrefix);

		if (rank == 0) {

			std::string xdmfName = prefix + ".xdmf";

			std::ofstream(xdmfName.c_str(), std::ios::app).close(); // Create the file (if it does not exist)
			m_xdmfFile.open(xdmfName.c_str());

			m_hdfFilename = prefix;
			// Remove all directories from prefix
			size_t pos = m_hdfFilename.find_last_of('/');
			if (pos != std::string::npos)
				m_hdfFilename = m_hdfFilename.substr(pos+1);
			m_hdfFilename += ".h5";
		}

		m_hdfFile = 0;
		m_hdfAccessList = H5P_DEFAULT;
		m_hdfVars = 0L;
		m_hdfVarMemSpace = 0;
#endif // USE_HDF
	}

	virtual ~XdmfWriter()
	{
#ifdef USE_HDF
		if (m_hdfVars)
			close();
#endif // USE_HDF
	}

	void init(unsigned int numCells, const unsigned int* cells, unsigned int numVertices, const double *vertices, bool useVertexFilter = true)
	{
#ifdef USE_HDF

		unsigned int offset = 0;
#ifdef PARALLEL
		// Apply vertex filter
		ParallelVertexFilter filter;
		if (useVertexFilter) {
			// Filter duplicate vertices
			filter.filter(numVertices, vertices);
			vertices = filter.localVertices();
			numVertices = filter.numLocalVertices();
		} else {
			// No vertex filter -> just get the offset we should at
			offset = numVertices;
			MPI_Scan(MPI_IN_PLACE, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
			offset -= numVertices;
		}
#endif // PARALLEL

		// Add vertex offset to all cells and convert to unsigned long
		unsigned long *h5Cells = new unsigned long[numCells * topoTypeSize()];
#ifdef PARALLEL
		if (useVertexFilter) {
#ifdef _OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (size_t i = 0; i < numCells*topoTypeSize(); i++)
				h5Cells[i] = filter.globalIds()[cells[i]];
		} else
#endif // PARALLEL
		{
#ifdef _OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (size_t i = 0; i < numCells*topoTypeSize(); i++)
				h5Cells[i] = cells[i] + offset;
		}

		// Initialize the XDMF file
		unsigned long totalSize[2] = {numCells, numVertices};
#ifdef PARALLEL
		MPI_Allreduce(MPI_IN_PLACE, totalSize, 2, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif // PARALLEL

		if (m_rank == 0) {
			m_xdmfFile << "<?xml version=\"1.0\" ?>" << std::endl
					<< "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl
					<< "<Xdmf Version=\"2.0\">" << std::endl
					<< " <Domain>" << std::endl;
			m_xdmfFile << "  <Topology TopologyType=\"" << topoTypeName() << "\" NumberOfElements=\"" << totalSize[0] << "\">" << std::endl
					<< "   <DataItem NumberType=\"UInt\" Precision=\"8\" Format=\"HDF\" Dimensions=\""
						<< totalSize[0] << " " << topoTypeSize() << "\">"
					<< m_hdfFilename << ":/connect"
					<< "</DataItem>" << std::endl
					<< "  </Topology>" << std::endl;
			m_xdmfFile << "  <Geometry name=\"geo\" GeometryType=\"XYZ\" NumberOfElements=\"" << totalSize[1] << "\">" << std::endl
					<< "   <DataItem NumberType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << totalSize[1] << " 3\">"
					<< m_hdfFilename << ":/geometry"
					<< "</DataItem>" << std::endl
					<< "  </Geometry>" << std::endl;

			m_xdmfFile << "  <DataItem NumberType=\"UInt\" Precision=\"4\" Format=\"HDF\" Dimensions=\"" << totalSize[0] << "\">"
					<< m_hdfFilename << ":/partition"
					<< "</DataItem>" << std::endl;

			m_timeDimPos = new size_t[m_variableNames.size()];
			for (size_t i = 0; i < m_variableNames.size(); i++) {
				m_xdmfFile << "  <DataItem NumberType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"";
				m_timeDimPos[i] = m_xdmfFile.tellp();
				m_xdmfFile << std::setw(MAX_TIMESTEP_SPACE) << m_timestep << ' ' << totalSize[0] << "\">"
						<< m_hdfFilename << ":/" << m_variableNames[i]
						<< "</DataItem>" << std::endl;
			}

			m_xdmfFile << "  <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

			if (m_timestep == 0)
				closeXdmf();
			else {
				// Jump the correct position in the file
				std::ostringstream tStartStream;
				timeStepStartXdmf(m_timestep-1, tStartStream);
				std::string tStart = tStartStream.str();

				// Find beginning of the (correct) time step
				std::string line;
				while (getline(m_xdmfFile, line)) {
					if (line.find(tStart) != std::string::npos)
						break;
				}
				if (!m_xdmfFile)
					logError() << "Unable to find time step for appending";

				// Find end of this time step
				while (getline(m_xdmfFile, line)) {
					if (line.find("</Grid>") != std::string::npos)
						break;
				}
			}
		}

		// Create partition information
		unsigned int *partInfo = new unsigned int[numCells];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numCells; i++)
			partInfo[i] = m_rank;

#ifdef PARALLEL
		// Create block buffer
		unsigned long blockSize = utils::Env::get<unsigned long>("XDMFWRITER_BLOCK_SIZE", 1);
		if (blockSize > 1) {
			m_blockBuffer.init(numCells, MPI_DOUBLE, blockSize);

			MPI_Comm_split(MPI_COMM_WORLD, m_blockBuffer.count() > 0 ? 1 : MPI_UNDEFINED, 0, &m_comm);
		}

		// Exchange cells and vertices
		// Be careful with allocation/deallocation!!!
		unsigned long *blockedCells = 0L;
		double *blockedVertices = 0L;
		unsigned int *blockedPartInfo = 0L;
		if (m_blockBuffer.isInitialized()) {
			numCells = m_blockBuffer.count();

			if (m_timestep == 0) {
				// Cells, vertices and partition info only needs to be exchanged
				// when creating a new file
				blockedCells = new unsigned long[numCells * topoTypeSize()];
				m_blockBuffer.exchange(h5Cells, MPI_UNSIGNED_LONG, topoTypeSize(), blockedCells);


				blockedVertices = m_blockBuffer.exchangeAny(vertices, MPI_DOUBLE, 3,
						numVertices, numVertices);

				if (numCells > 0)
					blockedPartInfo = new unsigned int[numCells];
				m_blockBuffer.exchange(partInfo, MPI_UNSIGNED, 1, blockedPartInfo);

				// Overwrite pointers
				delete [] h5Cells;
				h5Cells = blockedCells;

				vertices = blockedVertices;

				delete [] partInfo;
				partInfo = blockedPartInfo;
			}

			// Allocate memory for data exchange
			if (numCells > 0)
				m_blocks = new double[numCells];
		}
#endif // PARALLEL

		// Create and initialize the HDF5 file
		if (m_blockBuffer.count() > 0) {
			std::string hdfName = m_outputPrefix + ".h5";

			// Create the file
			hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
			checkH5Err(h5plist);
			checkH5Err(H5Pset_libver_bounds(h5plist, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
			checkH5Err(H5Pset_meta_block_size(h5plist, 1024*1024));
			hsize_t align = utils::Env::get<hsize_t>("XDMFWRITER_ALIGNMENT", 0);
			if (align > 0)
				checkH5Err(H5Pset_alignment(h5plist, 1, align));
#ifdef PARALLEL
			checkH5Err(H5Pset_fapl_mpio(h5plist, m_comm, MPI_INFO_NULL));
#endif // PARALLEL

			if (m_timestep == 0) {
				struct stat statBuffer;
				if (m_rank == 0 && stat(hdfName.c_str(), &statBuffer) == 0) {
					logWarning() << "HDF5 output file already exists. Creating backup.";
					rename(hdfName.c_str(), (hdfName + ".bak").c_str());
				}

#ifdef PARALLEL
				// Make sure the file is moved before continuing
				MPI_Barrier(m_comm);
#endif // PARALLEL

				m_hdfFile = H5Fcreate(hdfName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, h5plist);
			} else
				m_hdfFile = H5Fopen(hdfName.c_str(), H5F_ACC_RDWR, h5plist);
			checkH5Err(m_hdfFile);

			checkH5Err(H5Pclose(h5plist));

			// Compute the offsets where we should start in HDF5 file
			unsigned long offsets[2] = {numCells, numVertices};
#ifdef PARALLEL
			MPI_Scan(MPI_IN_PLACE, offsets, 2, MPI_UNSIGNED_LONG, MPI_SUM, m_comm);
#endif // PARALLEL
			offsets[0] -= numCells;
			offsets[1] -= numVertices;

			// Allocate array for the variable ids
			m_hdfVars = new hid_t[m_variableNames.size()];

			// Parallel access for all datasets
#ifdef PARALLEL
			m_hdfAccessList = H5Pcreate(H5P_DATASET_XFER);
			checkH5Err(m_hdfAccessList);
			checkH5Err(H5Pset_dxpl_mpio(m_hdfAccessList, H5FD_MPIO_COLLECTIVE));
#endif // PARALLEL

			if (m_timestep == 0) {
				// Create connect dataset
				hsize_t connectDims[2] = {totalSize[0], topoTypeSize()};
				hid_t h5ConnectSpace = H5Screate_simple(2, connectDims, 0L);
				checkH5Err(h5ConnectSpace);
				hid_t h5Connect = H5Dcreate(m_hdfFile, "/connect", H5T_STD_U64LE, h5ConnectSpace,
						H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
				checkH5Err(h5Connect);

				// Create geometry dataset
				hsize_t geometryDims[2] = {totalSize[1], 3};
				hid_t h5GeometrySpace = H5Screate_simple(2, geometryDims, 0L);
				checkH5Err(h5GeometrySpace);
				hid_t h5Geometry = H5Dcreate(m_hdfFile, "/geometry", H5T_IEEE_F64LE, h5GeometrySpace,
						H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
				checkH5Err(h5Geometry);

				// Create partition dataset
				hsize_t partDim = totalSize[0];
				hid_t h5PartitionSpace = H5Screate_simple(1, &partDim, 0L);
				checkH5Err(h5PartitionSpace);
				hid_t h5Partition = H5Dcreate(m_hdfFile, "/partition", H5T_STD_U32LE, h5PartitionSpace,
						H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
				checkH5Err(h5Partition);

				// Create variable datasets
				hsize_t varChunkDims[2] = {
						utils::Env::get<hsize_t>("XDMFWRITER_TIME_CHUNK_SIZE", 1),
						utils::Env::get<hsize_t>("XDMFWRITER_ELEMENT_CHUNK_SIZE", 0)};
				if (varChunkDims[1] == 0)
					// 0 elements -> all elements
					varChunkDims[1] = totalSize[0];
				// TODO add additional check for the chunk size
				hid_t h5pCreate = H5Pcreate(H5P_DATASET_CREATE);
				checkH5Err(h5pCreate);
				checkH5Err(H5Pset_chunk(h5pCreate, 2, varChunkDims));

				for (size_t i = 0; i < m_variableNames.size(); i++) {
					std::string varName("/");
					varName += m_variableNames[i];

					hsize_t varDims[2] = {0, totalSize[0]};
					hsize_t varDimsMax[2] = {H5S_UNLIMITED, totalSize[0]};
					hid_t h5VarSpace = H5Screate_simple(2, varDims, varDimsMax);
					checkH5Err(h5VarSpace);

					m_hdfVars[i] = H5Dcreate(m_hdfFile, varName.c_str(), H5T_IEEE_F64LE, h5VarSpace,
							H5P_DEFAULT, h5pCreate, H5P_DEFAULT);
					checkH5Err(m_hdfVars[i]);

					checkH5Err(H5Sclose(h5VarSpace));
				}
				checkH5Err(H5Pclose(h5pCreate));

				// Write connectivity
				hsize_t connectWriteStart[2] = {offsets[0], 0};
				hsize_t connectWriteCount[2] = {numCells, topoTypeSize()};
				hid_t h5MemSpace = H5Screate_simple(2, connectWriteCount, 0L);
				checkH5Err(h5MemSpace);
				checkH5Err(H5Sselect_hyperslab(h5ConnectSpace, H5S_SELECT_SET, connectWriteStart, 0L, connectWriteCount, 0L));

				checkH5Err(H5Dwrite(h5Connect, H5T_NATIVE_ULONG, h5MemSpace, h5ConnectSpace, m_hdfAccessList, h5Cells));
				checkH5Err(H5Sclose(h5MemSpace));

				// Write geometry
				hsize_t geometryWriteStart[2] = {offsets[1], 0};
				hsize_t geometryWriteCount[2] = {numVertices, 3};
				h5MemSpace = H5Screate_simple(2, geometryWriteCount, 0L);
				checkH5Err(h5MemSpace);
				checkH5Err(H5Sselect_hyperslab(h5GeometrySpace, H5S_SELECT_SET, geometryWriteStart, 0L, geometryWriteCount, 0L));

				checkH5Err(H5Dwrite(h5Geometry, H5T_NATIVE_DOUBLE, h5MemSpace, h5GeometrySpace, m_hdfAccessList, vertices));
				checkH5Err(H5Sclose(h5MemSpace));

#ifdef PARALLEL
				BlockBuffer::free(blockedVertices);
#endif // PARALLEL

				// Write partition information
				hsize_t partitionWriteStart = offsets[0];
				hsize_t partitionWriteCount = numCells;
				h5MemSpace = H5Screate_simple(1, &partitionWriteCount, 0L);
				checkH5Err(h5MemSpace);
				checkH5Err(H5Sselect_hyperslab(h5PartitionSpace, H5S_SELECT_SET, &partitionWriteStart, 0L, &partitionWriteCount, 0L));

				checkH5Err(H5Dwrite(h5Partition, H5T_NATIVE_UINT, h5MemSpace, h5PartitionSpace, m_hdfAccessList, partInfo));
				checkH5Err(H5Sclose(h5MemSpace));

				delete [] partInfo;

				// Close all datasets we only write once
				checkH5Err(H5Dclose(h5Connect));
				checkH5Err(H5Sclose(h5ConnectSpace));
				checkH5Err(H5Dclose(h5Geometry));
				checkH5Err(H5Sclose(h5GeometrySpace));
				checkH5Err(H5Dclose(h5Partition));
				checkH5Err(H5Sclose(h5PartitionSpace));

				checkH5Err(H5Fflush(m_hdfFile, H5F_SCOPE_GLOBAL));
			} else {
				// Get variable datasets

				for (size_t i = 0; i < m_variableNames.size(); i++) {
					std::string varName("/");
					varName += m_variableNames[i];

					m_hdfVars[i] = H5Dopen(m_hdfFile, varName.c_str(), H5P_DEFAULT);
					checkH5Err(m_hdfVars[i]);
				}
			}

			// Memory space we use for writing one variable at one time step
			hsize_t writeCount[2] = {1, numCells};
			m_hdfVarMemSpace = H5Screate_simple(2, writeCount, 0L);
			checkH5Err(m_hdfVarMemSpace);

			// Save values we require for resizing the variables and setting the local hyperslab
			m_totalCells = totalSize[0];
			m_localCells = numCells;
			m_offsetCells = offsets[0];

			// Get flush interval
			m_flushInterval = utils::Env::get<unsigned int>("XDMFWRITER_FLUSH_INTERVAL", 1);
		}

		delete [] h5Cells;
#endif // USE_HDF
	}

	/**
	 * Closes the HDF5 file (should be done before MPI_Finalize is called)
	 */
	void close()
	{
#ifdef USE_HDF
		if (m_blockBuffer.count() > 0) {
			for (size_t i = 0; i < m_variableNames.size(); i++)
				checkH5Err(H5Dclose(m_hdfVars[i]));
			checkH5Err(H5Sclose(m_hdfVarMemSpace));
			checkH5Err(H5Pclose(m_hdfAccessList));
			checkH5Err(H5Fclose(m_hdfFile));

#ifdef PARALLEL
			if (m_blockBuffer.isInitialized())
				MPI_Comm_free(&m_comm);
#endif // PARALLEL
		}

		delete [] m_timeDimPos;
		delete [] m_hdfVars;
		m_hdfVars = 0L; // Indicates closed file

		delete [] m_blocks;
#endif // USE_HDF
	}

	/**
	 * Add a new output time step
	 */
	void addTimeStep(double time)
	{
#ifdef USE_HDF
		if (m_rank == 0) {
			m_xdmfFile << "   ";
			timeStepStartXdmf(m_timestep, m_xdmfFile);
			m_xdmfFile << std::endl;
			m_xdmfFile << "    <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>" << std::endl
					<< "    <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>" << std::endl
					<< "    <Time Value=\"" << time << "\"/>" << std::endl;
			m_xdmfFile << "    <Attribute Name= \"partition\" Center=\"Cell\">" << std::endl
					<< "     <DataItem Reference=\"/Xdmf/Domain/DataItem[1]\"/>" << std::endl
					<< "    </Attribute>" << std::endl;
			for (size_t i = 0; i < m_variableNames.size(); i++) {
				m_xdmfFile << "    <Attribute Name= \"" << m_variableNames[i] << "\" Center=\"Cell\">" << std::endl
						<< "     <DataItem ItemType=\"HyperSlab\" Dimensions=\"" << m_totalCells << "\">" << std::endl
						<< "      <DataItem NumberType=\"UInt\" Precision=\"8\" Format=\"XML\" Dimensions=\"3 2\">"
						<< m_timestep << " 0 1 1 1 " << m_totalCells << "</DataItem>" << std::endl
						<< "      <DataItem Reference=\"/Xdmf/Domain/DataItem[" << (i+2) << "]\"/>" << std::endl // PartInfo + 1 based index
						<< "     </DataItem>" << std::endl
						<< "    </Attribute>" << std::endl;
			}
			m_xdmfFile << "   </Grid>" << std::endl;

			// Update total steps information
			size_t pos = m_xdmfFile.tellp();
			for (size_t i = 0; i < m_variableNames.size(); i++) {
				m_xdmfFile.seekp(m_timeDimPos[i]);
				m_xdmfFile << std::setw(MAX_TIMESTEP_SPACE) << (m_timestep+1);
			}
			m_xdmfFile.seekp(pos);

			closeXdmf();
		}

		m_timestep++;
#endif // USE_HDF
	}

	/**
	 * Write data for one variable at the current time step
	 *
	 * @param id The number of the variable that should be written
	 */
	void writeData(unsigned int id, const double *data)
	{
		EPIK_TRACER("XDMFWriter_writeData");
		SCOREP_USER_REGION("XDMFWriter_writeData", SCOREP_USER_REGION_TYPE_FUNCTION);

#ifdef USE_HDF
#ifdef PARALLEL
		if (m_blockBuffer.isInitialized()) {
			m_blockBuffer.exchange(data, m_blocks);
			data = m_blocks;
		}
#endif // PARALLEL

		if (m_blockBuffer.count() > 0) {
			hsize_t extent[2] = {m_timestep, m_totalCells};
			checkH5Err(H5Dset_extent(m_hdfVars[id], extent));

			hid_t h5VarSpace = H5Dget_space(m_hdfVars[id]);
			checkH5Err(h5VarSpace);
			hsize_t writeStart[2] = {m_timestep-1, m_offsetCells}; // We already increment m_timestep
			hsize_t writeCount[2] = {1, m_localCells};
			checkH5Err(H5Sselect_hyperslab(h5VarSpace, H5S_SELECT_SET, writeStart, 0L, writeCount, 0L));

			checkH5Err(H5Dwrite(m_hdfVars[id], H5T_NATIVE_DOUBLE, m_hdfVarMemSpace,
					h5VarSpace, m_hdfAccessList, data));

			checkH5Err(H5Sclose(h5VarSpace));
		}
#endif // USE_HDF
	}

	/**
	 * Flushes the data to disk
	 */
	void flush()
	{
		EPIK_TRACER("XDMFWriter_flush");
		SCOREP_USER_REGION("XDMFWriter_flush", SCOREP_USER_REGION_TYPE_FUNCTION);

#ifdef USE_HDF
		if (m_blockBuffer.count() > 0) {
			if (m_timestep % m_flushInterval == 0)
				checkH5Err(H5Fflush(m_hdfFile, H5F_SCOPE_GLOBAL));
		}
#endif // USE_HDF
	}

	/**
	 * @return The current time step of the output file
	 */
	unsigned int timestep() const
	{
		return m_timestep;
	}

private:
	void closeXdmf()
	{
#ifdef USE_HDF
		size_t contPos = m_xdmfFile.tellp();
		m_xdmfFile << "  </Grid>" << std::endl
				<< " </Domain>" << std::endl
				<< "</Xdmf>" << std::endl;
		m_xdmfFile.seekp(contPos);
#endif // USE_HDF
	}

	/**
	 * @return Name of the topology type in the XDMF file
	 */
	const char* topoTypeName() const;

	/**
	 * @return Number of vertices of the topology type
	 */
	unsigned int topoTypeSize() const;

private:
	/**
	 * Write the beginning of a time step to the stream
	 */
	static void timeStepStartXdmf(unsigned int timestep, std::ostream &s)
	{
		s << "<Grid Name=\"step_" << std::setw(MAX_TIMESTEP_SPACE) << std::setfill('0') << timestep << std::setfill(' ')
				<< "\" GridType=\"Uniform\">";
	}

	template<typename T>
	static void checkH5Err(T status)
	{
		if (status < 0)
			logError() << "An HDF5 error occurred in the XDMF writer";
	}

private:
	static const unsigned int MAX_TIMESTEP_SPACE = 12;
};

template<> inline
const char* XdmfWriter<TRIANGLE>::topoTypeName() const
{
	return "Triangle";
}

template<> inline
const char* XdmfWriter<TETRAHEDRON>::topoTypeName() const
{
	return "Tetrahedron";
}

template<> inline
unsigned int XdmfWriter<TRIANGLE>::topoTypeSize() const
{
	return 3;
}

template<> inline
unsigned int XdmfWriter<TETRAHEDRON>::topoTypeSize() const
{
	return 4;
}

}

#endif // XDMF_WRITER_H
