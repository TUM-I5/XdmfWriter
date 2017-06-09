/**
 * @file
 *  This file is part of XdmfWriter
 *
 * @author Sebastian Rettenberger <sebastian.rettenberger@tum.de>
 *
 * @copyright Copyright (c) 2017, Technische Universitaet Muenchen.
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

#ifndef XDMFWRITER_INDEXSORT_H
#define XDMFWRITER_INDEXSORT_H

#include <algorithm>

namespace xdmfwriter
{

namespace internal
{

enum SortType
{
	STABLE,
	UNSTABLE
};

/**
 * Compares 3D-vertex indices according to the vertices
 */
template<typename T>
class IndexedVertexComparator
{
private:
	const T *m_vertices;

public:
	IndexedVertexComparator(const T *vertices)
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

template<SortType, typename T>
struct RunSort
{
	void operator()(unsigned int* first, unsigned int* last, const IndexedVertexComparator<T> &comparator);
};

template<typename T>
struct RunSort<STABLE, T>
{
	void operator()(unsigned int* first, unsigned int* last, const IndexedVertexComparator<T> &comparator)
	{
		std::stable_sort(first, last, comparator);
	}
};

template<typename T>
struct RunSort<UNSTABLE, T>
{
	void operator()(unsigned int* first, unsigned int* last, const IndexedVertexComparator<T> &comparator)
	{
		std::sort(first, last, comparator);
	}
};

/**
 * Creates a sorted index
 */
template<SortType S, typename T>
class IndexSort
{
public:
	static void sort(const T *vertices, unsigned int numVertices, unsigned int *sortedIndices)
	{
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (unsigned int i = 0; i < numVertices; i++)
			sortedIndices[i] = i;

		IndexedVertexComparator<T> comparator(vertices);

		RunSort<S, T> runner;
		runner(sortedIndices, sortedIndices+numVertices, comparator);
	}
};

}

}

#endif // XDMFWRITER_INDEXSORT_H