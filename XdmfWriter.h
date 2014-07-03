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

#ifndef XDMF_WRITER_H
#define XDMF_WRITER_H

#ifdef PARALLEL
#include <mpi.h>
#endif // PARALLEL

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#ifdef USE_HDF
#include <hdf5.h>
#endif // USE_HDF

#include "utils/logger.h"

#include "epik_wrapper.h"
#ifdef PARALLEL
#include "ParallelVertexFilter.h"
#endif // PARALLEL

/**
 * Writes data in XDMF format
 */
class XdmfWriter
{
private:
	int m_rank;

	std::string m_outputPrefix;

	std::ofstream m_xdmfFile;

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

	/** Offsets in the XDMF file which describe the time dimension size */
	size_t *m_timeDimPos;

	/** Output step counter */
	unsigned int m_timestep;

public:
	XdmfWriter(int rank, const char* outputPrefix, const std::vector<const char*> &variableNames)
		: m_rank(rank), m_outputPrefix(outputPrefix), m_variableNames(variableNames),
		  m_totalCells(0), m_localCells(0), m_offsetCells(0), m_timeDimPos(0L), m_timestep(0)
	{
		std::string prefix(outputPrefix);

		if (rank == 0) {

			std::string xdmfName = prefix + ".xdmf";

			m_xdmfFile.open(xdmfName.c_str());

			m_hdfFilename = prefix;
			// Remove all directories from prefix
			size_t pos = m_hdfFilename.find_last_of('/');
			if (pos != std::string::npos)
				m_hdfFilename = m_hdfFilename.substr(pos+1);
			m_hdfFilename += ".h5";
		}

#ifdef USE_HDF
		m_hdfFile = 0;
		m_hdfAccessList = H5P_DEFAULT;
		m_hdfVars = 0L;
		m_hdfVarMemSpace = 0;
#endif // USE_HDF
	}

	virtual ~XdmfWriter()
	{
		delete [] m_timeDimPos;
#ifdef USE_HDF
		delete [] m_hdfVars;
#endif // USE_HDF
	}

	void init(unsigned int numCells, const unsigned int* cells, unsigned int numVertices, const double *vertices, bool useVertexFilter = true)
	{
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
			m_xdmfFile << "  <Topology TopologyType=\"Tetrahedron\" NumberOfElements=\"" << totalSize[0] << "\">" << std::endl
					<< "   <DataItem NumberType=\"UInt\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << totalSize[0] << " 4\">"
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
				m_xdmfFile << std::setw(MAX_TIMESTEP_SPACE) << 0 << ' ' << totalSize[0] << "\">"
						<< m_hdfFilename << ":/" << m_variableNames[i]
						<< "</DataItem>" << std::endl;
			}

			m_xdmfFile << "  <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;
			closeXdmf();
		}

		// Create and initialize the HDF5 file
#ifdef USE_HDF
		std::string hdfName = m_outputPrefix + ".h5";

		// Create the file
#ifdef PARALLEL
		hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
		H5Pset_fapl_mpio(h5plist, MPI_COMM_WORLD, MPI_INFO_NULL);
#else // PARALLEL
		hid_t h5plist = H5P_DEFAULT;
#endif // PARALLEL

		m_hdfFile = H5Fcreate(hdfName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, h5plist);

#ifdef PARALLEL
		H5Pclose(h5plist);
#endif // PARALLEL
#endif // USE_HDF

#ifdef PARALLEL
		ParallelVertexFilter filter;
		if (useVertexFilter) {
			// Filter duplicate vertices
			filter.filter(numVertices, vertices);
			vertices = filter.localVertices();
			numVertices = filter.numLocalVertices();
		}
#endif // PARALLEL

		// Compute the offsets where we should start in HDF5 file
		unsigned long offsets[2] = {numCells, numVertices};
#ifdef PARALLEL
		MPI_Scan(MPI_IN_PLACE, offsets, 2, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif // PARALLEL
		offsets[0] -= numCells;
		offsets[1] -= numVertices;

#ifdef USE_HDF
		// Create connect dataset
		hsize_t connectDims[2] = {totalSize[0], 4};
		hid_t h5ConnectSpace = H5Screate_simple(2, connectDims, 0L);
		hid_t h5Connect = H5Dcreate(m_hdfFile, "/connect", H5T_STD_U64LE, h5ConnectSpace,
				H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		// Create geometry dataset
		hsize_t geometryDims[2] = {totalSize[1], 3};
		hid_t h5GeometrySpace = H5Screate_simple(2, geometryDims, 0L);
		hid_t h5Geometry = H5Dcreate(m_hdfFile, "/geometry", H5T_IEEE_F64LE, h5GeometrySpace,
				H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		// Create partition dataset
		hsize_t partDim = totalSize[0];
		hid_t h5PartitionSpace = H5Screate_simple(1, &partDim, 0L);
		hid_t h5Partition = H5Dcreate(m_hdfFile, "/partition", H5T_STD_U32LE, h5PartitionSpace,
				H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		// Create variable datasets
		m_hdfVars = new hid_t[m_variableNames.size()];

		hsize_t varChunkDims[2] = {1, totalSize[0]};
		hid_t h5pCreate = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_chunk(h5pCreate, 2, varChunkDims);

		for (size_t i = 0; i < m_variableNames.size(); i++) {
			hsize_t varDims[2] = {0, totalSize[0]};
			hsize_t varDimsMax[2] = {H5S_UNLIMITED, totalSize[0]};
			hid_t h5VarSpace = H5Screate_simple(2, varDims, varDimsMax);
			std::string varName("/");
			varName += m_variableNames[i];

			m_hdfVars[i] = H5Dcreate(m_hdfFile, varName.c_str(), H5T_IEEE_F64LE, h5VarSpace,
					H5P_DEFAULT, h5pCreate, H5P_DEFAULT);

			H5Sclose(h5VarSpace);
		}
		H5Pclose(h5pCreate);

		// Parallel access for all datasets
#ifdef PARALLEL
		m_hdfAccessList = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(m_hdfAccessList, H5FD_MPIO_COLLECTIVE);
#endif // PARALLEL

		// Write connectivity
		hsize_t connectWriteStart[2] = {offsets[0], 0};
		hsize_t connectWriteCount[2] = {numCells, 4};
		hid_t h5MemSpace = H5Screate_simple(2, connectWriteCount, 0L);
		H5Sselect_hyperslab(h5ConnectSpace, H5S_SELECT_SET, connectWriteStart, 0L, connectWriteCount, 0L);

		// Add vertex offset to all cells
		unsigned long *h5Cells = new unsigned long[numCells * 4];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (size_t i = 0; i < numCells*4; i++) {
#ifdef PARALLEL
			if (useVertexFilter)
				h5Cells[i] = filter.globalIds()[cells[i]];
			else
#endif // PARALLEL
				h5Cells[i] = cells[i] + offsets[1];
		}
		herr_t h5status = H5Dwrite(h5Connect, H5T_NATIVE_ULONG, h5MemSpace, h5ConnectSpace, m_hdfAccessList, h5Cells);
		delete [] h5Cells;
		H5Sclose(h5MemSpace);

		// Write geometry
		hsize_t geometryWriteStart[2] = {offsets[1], 0};
		hsize_t geometryWriteCount[2] = {numVertices, 3};
		h5MemSpace = H5Screate_simple(2, geometryWriteCount, 0L);
		H5Sselect_hyperslab(h5GeometrySpace, H5S_SELECT_SET, geometryWriteStart, 0L, geometryWriteCount, 0L);

		h5status = H5Dwrite(h5Geometry, H5T_NATIVE_DOUBLE, h5MemSpace, h5GeometrySpace, m_hdfAccessList, vertices);
		H5Sclose(h5MemSpace);

		// Write partition information
		hsize_t partitionWriteStart = offsets[0];
		hsize_t partitionWriteCount = numCells;
		h5MemSpace = H5Screate_simple(1, &partitionWriteCount, 0L);
		H5Sselect_hyperslab(h5PartitionSpace, H5S_SELECT_SET, &partitionWriteStart, 0L, &partitionWriteCount, 0L);

		unsigned int *h5PartInfo = new unsigned int[numCells];
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numCells; i++)
			h5PartInfo[i] = m_rank;
		h5status = H5Dwrite(h5Partition, H5T_NATIVE_UINT, h5MemSpace, h5PartitionSpace, m_hdfAccessList, h5PartInfo);
		H5Sclose(h5MemSpace);

		// Close all datasets we only write once
		H5Dclose(h5Connect);
		H5Sclose(h5ConnectSpace);
		H5Dclose(h5Geometry);
		H5Sclose(h5GeometrySpace);
		H5Dclose(h5Partition);
		H5Sclose(h5PartitionSpace);

		h5status = H5Fflush(m_hdfFile, H5F_SCOPE_GLOBAL);

		// Memory space we use for writing one variable at one time step
		hsize_t writeCount[2] = {1, numCells};
		m_hdfVarMemSpace = H5Screate_simple(2, writeCount, 0L);
#endif // USE_HDF

		// Save values we require for resizing the variables and setting the local hyperslab
		m_totalCells = totalSize[0];
		m_localCells = numCells;
		m_offsetCells = offsets[0];
	}

	/**
	 * Closes the HDF5 file (should be done before MPI_Finalize is called)
	 */
	void close()
	{
#ifdef USE_HDF
		for (size_t i = 0; i < m_variableNames.size(); i++)
			H5Dclose(m_hdfVars[i]);
		H5Sclose(m_hdfVarMemSpace);
		H5Pclose(m_hdfAccessList);
		H5Fclose(m_hdfFile);
#endif // USE_HDF
	}

	/**
	 * Add a new output time step
	 */
	void addTimeStep(double time)
	{
		if (m_rank == 0) {
			m_xdmfFile << "   <Grid Name=\"step_" << std::setw(MAX_TIMESTEP_SPACE) << std::setfill('0') << m_timestep << std::setfill(' ')
					<< "\" GridType=\"Uniform\">" << std::endl
					<< "    <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>" << std::endl
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
	}

	/**
	 * Write data for one variable at the current time step
	 *
	 * @param id The number of the variable that should be written
	 */
	void writeData(unsigned int id, const double *data)
	{
		EPIK_TRACER("XDMFWriter_writeData");

#ifdef USE_HDF
		hsize_t extent[2] = {m_timestep, m_totalCells};
		H5Dset_extent(m_hdfVars[id], extent);

		hid_t h5VarSpace = H5Dget_space(m_hdfVars[id]);
		hsize_t writeStart[2] = {m_timestep-1, m_offsetCells}; // We already increment m_timestep
		hsize_t writeCount[2] = {1, m_localCells};
		H5Sselect_hyperslab(h5VarSpace, H5S_SELECT_SET, writeStart, 0L, writeCount, 0L);

		herr_t h5status = H5Dwrite(m_hdfVars[id], H5T_NATIVE_DOUBLE, m_hdfVarMemSpace,
				h5VarSpace, m_hdfAccessList, data);

		H5Sclose(h5VarSpace);
#endif // USE_HDF
	}

	/**
	 * Flushes the data to disk
	 */
	void flush()
	{
		EPIK_TRACER("XDMFWriter_flush");

#ifdef USE_HDF
		herr_t status = H5Fflush(m_hdfFile, H5F_SCOPE_GLOBAL);
#endif // USE_HDF
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
	static const unsigned int MAX_TIMESTEP_SPACE = 12;
};

#endif // XDMF_WRITER_H
