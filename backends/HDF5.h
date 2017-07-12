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

#ifndef XDMF_WRITER_BACKENDS_HDF5_H
#define XDMF_WRITER_BACKENDS_HDF5_H

#include <algorithm>
#include <cassert>
#include <string>

#include <hdf5.h>

#include "utils/env.h"
#include "utils/stringutils.h"

#include "Base.h"
#ifdef USE_MPI
#include "MPIInfo.h"
#endif // USE_MPI

namespace xdmfwriter
{

namespace backends
{

#define checkH5Err(...) _checkH5Err(__VA_ARGS__, __FILE__, __LINE__)

template<typename T>
class HDF5 : public Base<T>
{
private:
	/** True, if the HDF5 file should be created, otherwise an existing file is opened */
	bool m_create;

	bool m_useCollective;

	hid_t m_hdfFile;

	hid_t m_hdfAccessList;

	/** Variable identifiers */
	hid_t *m_hdfVars;

public:
	HDF5()
		: Base<T>("HDF"),
		m_create(true),
		m_useCollective(true),
		m_hdfFile(-1),
		m_hdfAccessList(H5P_DEFAULT),
		m_hdfVars(0L)
	{
	}

	void open(const std::string &outputPrefix, const std::vector<VariableData> &variableData, bool create = true)
	{
		Base<T>::open(outputPrefix, variableData, create);

		// Check if we should use collective I/O
		m_useCollective = utils::Env::get<bool>("XDMFWRITER_COLLECTIVE", true);

		// Create a backup of the file
		if (create) {
			// Backup existing file
			Base<T>::backup(outputPrefix + fileExention());
#ifdef PARALLEL
			// Make sure the file is moved before continuing
			MPI_Barrier(Base<T>::comm());
#endif // PARALLEL
		}

		m_create = create;

		// Allocate array for variable ids
		m_hdfVars = new hid_t[variableData.size()];
		std::fill_n(m_hdfVars, variableData.size(), -1);

		// Create the access list
#ifdef USE_MPI
		if (m_useCollective) {
			m_hdfAccessList = H5Pcreate(H5P_DATASET_XFER);
			checkH5Err(m_hdfAccessList);
			checkH5Err(H5Pset_dxpl_mpio(m_hdfAccessList, H5FD_MPIO_COLLECTIVE));
		}
#endif // USE_MPI
	}

	void setMesh(unsigned int meshId,
		unsigned long totalElements, unsigned int localElements, unsigned long offset)
	{
#ifdef USE_MPI
		// Were we a writer rank before?
		bool wasWriter = Base<T>::localElements() > 0;
#endif // USE_MPI

		Base<T>::setMesh(totalElements, localElements, offset);

		// Close any open variables
		closeVars();

		bool isWriter = Base<T>::localElements() > 0;
#ifdef USE_MPI
		if (m_useCollective) {
			// We need to reopen the file
			int reopen = wasWriter != isWriter;
			MPI_Allreduce(MPI_IN_PLACE, &reopen, 1, MPI_INT, MPI_LOR, Base<T>::comm());

			// We need to reopen, close the old file
			if (reopen && m_hdfFile >= 0) {
				checkH5Err(H5Fclose(m_hdfFile));
				m_hdfFile = -1;
			}
		}
#endif // USE_MPI

		if (m_hdfFile < 0 && (!m_useCollective || isWriter)) {
			// We need to create/(re)open the HDF5 file

			hid_t h5plist = H5Pcreate(H5P_FILE_ACCESS);
			checkH5Err(h5plist);
			checkH5Err(H5Pset_libver_bounds(h5plist, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
			checkH5Err(H5Pset_meta_block_size(h5plist, 1024*1024));
			hsize_t align = utils::Env::get<hsize_t>("XDMFWRITER_ALIGNMENT", 0);
			if (align > 0)
				checkH5Err(H5Pset_alignment(h5plist, 1, align));
#ifdef USE_MPI
			if (m_useCollective)
				checkH5Err(H5Pset_fapl_mpio(h5plist, Base<T>::ioComm(), MPIInfo::get()));
			else
				checkH5Err(H5Pset_fapl_mpio(h5plist, Base<T>::comm(), MPIInfo::get()));
#endif // USE_MPI

			// Assemble filename
			std::string filename = Base<T>::pathPrefix() + fileExention();

			if (m_create)
				m_hdfFile = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, h5plist);
			else
				m_hdfFile = H5Fopen(filename.c_str(), H5F_ACC_RDWR, h5plist);
			checkH5Err(m_hdfFile);

			checkH5Err(H5Pclose(h5plist));
		}

		// No need to create a new HDF5 file for the next mesh
		m_create = false;

		if (m_useCollective && !isWriter)
			return;

		// Create the new group in the file
		std::string groupName = "mesh" + utils::StringUtils::toString(meshId);
		htri_t hasGroup = H5Lexists(m_hdfFile, groupName.c_str(), H5P_DEFAULT);
		checkH5Err(hasGroup);
		if (hasGroup) {
			// Open the existing group
			hid_t h5group = H5Gopen(m_hdfFile, groupName.c_str(), H5P_DEFAULT);

			// Get variable datasets
			for (unsigned int i = 0; i < Base<T>::variables().size(); i++) {
				const VariableData &variable = Base<T>::variables()[i];

				m_hdfVars[i] = H5Dopen(h5group, variable.name, H5P_DEFAULT);
				checkH5Err(m_hdfVars[i]);
			}

			// Close H5 group
			checkH5Err(H5Gclose(h5group));
		} else {
			// Create a new group
			hid_t h5group = H5Gcreate(m_hdfFile, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			checkH5Err(h5group);

			// Get the chunk size
			hsize_t varChunkDims[3] = {
				utils::Env::get<hsize_t>("XDMFWRITER_TIME_CHUNK_SIZE", 1),
				utils::Env::get<hsize_t>("XDMFWRITER_ELEMENT_CHUNK_SIZE", 0)
			};
			if (varChunkDims[1] == 0)
				// 0 elements -> all elements
				varChunkDims[1] = Base<T>::totalElements();
			// TODO add additional check for the chunk size

			// Create the variables in the group
			for (unsigned int i = 0; i < Base<T>::variables().size(); i++) {
				const VariableData &variable = Base<T>::variables()[i];

				int dimensions = 1 + (variable.hasTime ? 1 : 0) + (variable.count > 1 ? 1 : 0);
				hsize_t sizes[3] = {0, Base<T>::totalElements(), variable.count};
				hsize_t maxSizes[3] = {H5S_UNLIMITED, Base<T>::totalElements(), variable.count};

				// Create the data space
				hsize_t* _sizes = sizes;
				hsize_t* _maxSizes = maxSizes;
				if (!variable.hasTime) {
					_sizes++;
					_maxSizes++;
				}
				hid_t h5space = H5Screate_simple(dimensions, _sizes, _maxSizes);
				checkH5Err(h5space);

				// Create the dataset property list
				hid_t h5pCreate = H5P_DEFAULT;
				if (variable.hasTime) {
					assert(dimensions == 2);

					h5pCreate = H5Pcreate(H5P_DATASET_CREATE);
					checkH5Err(h5pCreate);
					checkH5Err(H5Pset_chunk(h5pCreate, 2, varChunkDims));
				}

				m_hdfVars[i] = H5Dcreate(h5group, variable.name, uniqueH5Type(variable.type),
					h5space, H5P_DEFAULT, h5pCreate, H5P_DEFAULT);
				checkH5Err(m_hdfVars[i]);

				if (h5pCreate != H5P_DEFAULT)
					checkH5Err(H5Pclose(h5pCreate));
				checkH5Err(H5Sclose(h5space));
			}

			// Close H5 group
			checkH5Err(H5Gclose(h5group));
		}
	}

	void flush()
	{
		if (m_useCollective && Base<T>::localElements() <= 0)
			return;

		checkH5Err(H5Fflush(m_hdfFile, H5F_SCOPE_GLOBAL));
	}

	/**
	 * Closes the HDF5 file (should be done before MPI_Finalize is called)
	 */
	void close()
	{
		Base<T>::close();

		closeVars();

		if (m_hdfAccessList != H5P_DEFAULT) {
			checkH5Err(H5Pclose(m_hdfAccessList));
			m_hdfAccessList = H5P_DEFAULT;
		}

		if (m_hdfFile >= 0) {
			checkH5Err(H5Fclose(m_hdfFile));
			m_hdfFile = -1;
		}

		delete [] m_hdfVars;
		m_hdfVars = 0L;
	}

	/**
	 * @param var
	 * @return The location name in the XDMF file for a data item
	 */
	std::string dataLocation(unsigned int meshId, const char* var) const
	{
		return Base<T>::dataLocation(meshId, var) + fileExention() + ":/mesh" + utils::StringUtils::toString(meshId) + "/" + var;
	}

protected:
	void write(unsigned int timestep, unsigned int id, const void* data)
	{
		if (m_useCollective && Base<T>::localElements() <= 0)
			// set_extent is a collective operation, in independent mode,
			// we need to call this on all ranks
			return;

		const VariableData &variable = Base<T>::variables()[id];

		// If we have time, increase the size of the variable
		if (variable.hasTime) {
			assert(variable.count == 1);

			hsize_t extent[2] = {timestep+1, Base<T>::totalElements()};
			checkH5Err(H5Dset_extent(m_hdfVars[id], extent));
		}

		if (Base<T>::localElements() <= 0)
			// Nothing to write
			return;

		// Select the file space
		hid_t h5VarSpace = H5Dget_space(m_hdfVars[id]);
		checkH5Err(h5VarSpace);
		hsize_t writeStart[3] = {timestep, Base<T>::offset(), 0};
		hsize_t writeCount[3] = {1, Base<T>::localElements(), variable.count};

		hsize_t* _writeStart = writeStart;
		hsize_t* _writeCount = writeCount;
		if (!variable.hasTime) {
			_writeStart++;
			_writeCount++;
		}

		checkH5Err(H5Sselect_hyperslab(h5VarSpace, H5S_SELECT_SET, _writeStart, 0L, _writeCount, 0L));

		// Create the memory space
		int dimensions = 1 + (variable.hasTime ? 1 : 0) + (variable.count > 1 ? 1 : 0);
		hid_t h5VarMemSpace = H5Screate_simple(dimensions, _writeCount, 0L);
		checkH5Err(h5VarMemSpace);

		// Do the actual writting
		checkH5Err(H5Dwrite(m_hdfVars[id], nativeH5Type(variable.type), h5VarMemSpace,
				h5VarSpace, m_hdfAccessList, data));

		checkH5Err(H5Sclose(h5VarSpace));
		checkH5Err(H5Sclose(h5VarMemSpace));
	}

private:
	/**
	 * Closes variables (but not the file itself)
	 */
	void closeVars()
	{
		if (m_hdfVars == 0L || m_hdfVars[0] < 0)
			return;

		for (unsigned int i = 0; i < Base<T>::variables().size(); i++) {
			checkH5Err(H5Dclose(m_hdfVars[i]));
			m_hdfVars[i] = -1;
		}
	}

private:
	/**
	 * @return The file extension for this backend
	 */
	static const char* fileExention()
	{
		return ".h5";
	}

	static hid_t uniqueH5Type(VariableType type)
	{
		switch (type) {
		case FLOAT:
			if (sizeof(T) == sizeof(float))
				return H5T_IEEE_F32LE;
			return H5T_IEEE_F64LE;
		case INT:
			return H5T_STD_U32LE;
		case UNSIGNED_LONG:
			return H5T_STD_U64LE;
		}

		return -1;
	}

	static hid_t nativeH5Type(VariableType type)
	{
		switch (type) {
		case FLOAT:
			if (sizeof(T) == sizeof(float))
				return H5T_NATIVE_FLOAT;
			return H5T_NATIVE_DOUBLE;
		case INT:
			return H5T_NATIVE_UINT;
		case UNSIGNED_LONG:
			return H5T_NATIVE_ULONG;
		}

		return -1;
	}

	template<typename TT>
	static void _checkH5Err(TT status, const char* file, int line)
	{
		if (status < 0)
			logError() << utils::nospace << "An HDF5 error occurred in the XDMF writer ("
				<< file << ": " << line << ")";
	}
};

#undef checkH5Err

}

}

#endif // XDMF_WRITER_BACKENDS_HDF5_H
