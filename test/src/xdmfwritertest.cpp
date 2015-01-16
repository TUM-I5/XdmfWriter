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

#include <mpi.h>

#include "XdmfWriter.h"
#include "utils/logger.h"

#include <cstdlib>

#define NUM_CELLS 10000
#define NUM_VERTICES 2000

static int rand_max(int max)
{
	return rand() % max;
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank, numProcs;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<const char*> variables;
	variables.push_back("var1");
	variables.push_back("var2");

	XdmfWriter writer(rank, argv[1], variables);

	srand(rank*1000);

	unsigned int numCells = rand_max(NUM_CELLS*0.001)-NUM_CELLS*0.0005 + NUM_CELLS;
	unsigned int numVertices = rand_max(NUM_VERTICES*0.001)-NUM_VERTICES*0.0005 + NUM_VERTICES;

	logInfo(rank) << "Generating random mesh";

	unsigned int *cells = new unsigned int[numCells * 4];
	double *vertices = new double[numVertices * 3];

	for (unsigned int i = 0; i < numCells*4; i++)
		cells[i] = rand_max(numVertices);
	for (unsigned int i = 0; i < numVertices*3; i++)
		vertices[i] = rand() / static_cast<double>(RAND_MAX);

	logInfo(rank) << "Initialize XDMF writer";

	writer.init(numCells, cells, numVertices, vertices, true);

	double *data = new double[numCells];

	for (unsigned int i = 0; i < 20; i++) {
		// Synchronize to measure performance of writing a time step
		MPI_Barrier(MPI_COMM_WORLD);

		logInfo(rank) << "Adding time step" << i << "to the output file";

		writer.addTimeStep(i);
		for (unsigned int j = 0; j < 2; j++) {
			for (unsigned int k = 0; k < numCells; k++)
				data[k] = rand();

			writer.writeData(j, data);
		}
		writer.flush();
	}

	delete [] data;

	writer.close();

	delete [] cells;
	delete [] vertices;

	MPI_Finalize();
	return 0;
}
