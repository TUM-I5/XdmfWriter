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

#ifndef XDMF_WRITER_BACKENDS_POSIX_H
#define XDMF_WRITER_BACKENDS_POSIX_H

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "utils/stringutils.h"

#include "Base.h"

namespace xdmfwriter
{

namespace backends
{

#define checkErr(...) _checkErr(__VA_ARGS__, __FILE__, __LINE__)

template<typename T>
class Posix : public Base<T>
{
private:
	/** File handles */
	int* m_fh;

public:
	Posix() : Base<T>("Binary"),
		m_fh(0L)
	{
	}

	void open(const std::string &outputPrefix, const std::vector<VariableData> &variableData, bool create = true)
	{
		Base<T>::open(outputPrefix, variableData, create);

		if (create) {
			if (Base<T>::rank() == 0) {
				Base<T>::backup(outputPrefix);
				checkErr(mkdir(outputPrefix.c_str(), 0755));
			}

#ifdef USE_MPI
			// Ensure that everybody sees the new directory
			MPI_Barrier(Base<T>::comm());
#endif // USE_MPI
		}
	}

	void setMesh(unsigned int meshId,
		unsigned long totalElements, unsigned int localElements, unsigned long offset)
	{
		Base<T>::setMesh(totalElements, localElements, offset, true);

		// Backup the old folder an create a new one
		std::string folder = Base<T>::pathPrefix() + "/mesh" + utils::StringUtils::toString(meshId);

		if (Base<T>::rank() == 0) {
			// Check if the folder still exists (if we run from a checkpoint)
			struct stat statBuffer;
			if (stat(folder.c_str(), &statBuffer) != 0)
				checkErr(mkdir(folder.c_str(), 0755));
		}

#ifdef USE_MPI
		// Ensure that everybody sees the change
		MPI_Barrier(Base<T>::comm());
#endif // USE_MPI

		if (m_fh)
			closeFiles();
		else
			m_fh = new int[Base<T>::variables().size()];

		if (Base<T>::localElements() > 0) {
			for (unsigned int i = 0; i < Base<T>::variables().size(); i++) {
				m_fh[i] = open(filename(Base<T>::pathPrefix(), meshId, Base<T>::variables()[i].name).c_str());
			}
		} else {
			delete [] m_fh;
			m_fh = 0L;
		}
	}

	void flush()
	{
		if (m_fh) {
			for (unsigned int i = 0; i < Base<T>::variables().size(); i++)
				checkErr(fsync(m_fh[i]));
		}
	}

	/**
	 * Closes the POSIX files
	 */
	void close()
	{
		Base<T>::close();

		closeFiles();
		delete [] m_fh;
		m_fh = 0L;
	}

	/**
	 * @param var
	 * @return The location name in the XDMF file for a data item
	 */
	std::string dataLocation(unsigned int meshId, const char* var) const
	{
		return filename(Base<T>::dataLocation(meshId, var), meshId, var);
	}

protected:
	void write(unsigned int timestep, unsigned int id, const void* data)
	{
		if (Base<T>::localElements() == 0)
			// Nothing to write
			return;

		size_t offset = 0;
		if (Base<T>::variables()[id].hasTime)
			offset = Base<T>::totalElements() * timestep;
		offset += Base<T>::offset();
		offset *=  Base<T>::variables()[id].count * Base<T>::variableSize(Base<T>::variables()[id].type);
		write(m_fh[id], data, offset,
			Base<T>::localElements() * Base<T>::variables()[id].count * Base<T>::variableSize(Base<T>::variables()[id].type));
	}

private:
	void closeFiles()
	{
		if (m_fh) {
			for (unsigned int i = 0; i < Base<T>::variables().size(); i++)
				checkErr(::close(m_fh[i]));
		}
	}

private:
	/**
	 * @return The file name for a given variable
	 */
	static std::string filename(const std::string &prefix, unsigned int meshId, const char* var)
	{
		return prefix + "/mesh" + utils::StringUtils::toString(meshId) + "/" + var + ".bin";
	}

	static int open(const char* filename)
	{
		int fh = open64(filename, O_WRONLY | O_CREAT,
				S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
		checkErr(fh);
		return fh;
	}

	static void write(int fh, const void* buffer, size_t offset, size_t size)
	{
		checkErr(lseek64(fh, offset, SEEK_SET));

		const char* buf = reinterpret_cast<const char*>(buffer);
		while (size > 0) {
			ssize_t written = ::write(fh, buf, size);
			if (written <= 0)
				checkErr(written, size);
			buf += written;
			size -= written;
		}
	}

	template<typename TT>
	static void _checkErr(TT ret, const char* file, int line)
	{
		if (ret < 0)
			logError() << utils::nospace
				<< "An POSIX error occurred in the XDMF writer (" << file << ": " << line << "): " << strerror(errno);
	}

	/**
	 * Can be used to check read/write errors
	 *
	 * @param ret The return value
	 * @param target The expected return value (> 0)
	 */
	template<typename TT, typename U>
	static void _checkErr(TT ret, U target, const char* file, int line)
	{
		_checkErr(ret, file, line);
		if (ret != static_cast<TT>(target))
			logError() << utils::nospace
				<< "Error in XDMF writer (" << file << ": " << line << "): "
				<< target << " bytes expected; " << ret << " bytes gotten";
	}
};

#undef checkErr

}

}

#endif // XDMF_WRITER_BACKENDS_POSIX_H
