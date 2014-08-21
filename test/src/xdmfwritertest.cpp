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

	// Synchronize to measure performance of writing a time step
	MPI_Barrier(MPI_COMM_WORLD);

	double *data = new double[numCells];

	for (unsigned int i = 0; i < 20; i++) {
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
