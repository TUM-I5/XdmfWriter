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

#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "utils/args.h"
#include "utils/logger.h"

#include "XdmfWriter.h"

/**
 * @param layout The layout of ranks
 * @param size The size of the mesh in each dimension
 */
static void generateMesh(const int rank,
		const int layout[3],
		const unsigned int size[3],
		unsigned int *numCells, unsigned int *numVertices, unsigned int* &cells, double* &vertices)
{
	const int myblock[3] = {rank % layout[0], (rank / layout[0]) % layout[1], (rank / (layout[0] * layout[1])) % layout[2]};

	unsigned int start[3];
	unsigned int mysize[3];
	for (unsigned int i = 0; i < 3; i++) {
		mysize[i] = (size[i] + layout[i] - 1) / layout[i];
		start[i] = mysize[i] * myblock[i];

		// Correct last row
		mysize[i] = std::min(mysize[i], size[i] - start[i]);
	}

	const double cellSize[3] = {1./size[0], 1./size[1], 1./size[2]};

	*numCells = mysize[0]*mysize[1]*mysize[2];
	*numVertices = (mysize[0]+1) * (mysize[1]+1) * (mysize[2]+1);;

	delete [] cells;
	delete [] vertices;
	cells = new unsigned int[*numCells * 8];
	vertices = new double[*numVertices * 3];

	for (unsigned int z = 0; z < mysize[2]; z++) {
		for (unsigned int y = 0; y < mysize[1]; y++) {
			for (unsigned int x = 0; x < mysize[0]; x++) {
				const unsigned int offsetVertex = x + (mysize[0]+1)*y + (mysize[0]+1)*(mysize[1]+1)*z;

				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8] = offsetVertex;
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 1] = offsetVertex + 1;
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 3] = offsetVertex + mysize[0]+1;
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 2] = offsetVertex + mysize[0]+1 + 1;
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 4] = offsetVertex + (mysize[0]+1)*(mysize[1]+1);
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 5] = offsetVertex + (mysize[0]+1)*(mysize[1]+1) + 1;
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 7] = offsetVertex + (mysize[0]+1)*(mysize[1]+1) + mysize[0]+1;
				cells[(x + y*mysize[0] + z*mysize[0]*mysize[1])*8 + 6] = offsetVertex + (mysize[0]+1)*(mysize[1]+1) + mysize[0]+1 + 1;
				assert(offsetVertex + (mysize[0]+1)*(mysize[1]+1) + mysize[1]+1 + 1 < *numVertices);
			}
		}
	}
	for (unsigned int z = 0; z < mysize[2]+1; z++) {
		for (unsigned int y = 0; y < mysize[1]+1; y++) {
			for (unsigned int x = 0; x < mysize[0]+1; x++) {
				(vertices)[3 * ((mysize[0]+1)*(mysize[1]+1)*z + (mysize[0]+1)*y + x)] = ((double)(x+start[0])) * cellSize[0];
				(vertices)[3 * ((mysize[0]+1)*(mysize[1]+1)*z + (mysize[0]+1)*y + x) + 1] = ((double)(y+start[1])) * cellSize[1];
				(vertices)[3 * ((mysize[0]+1)*(mysize[1]+1)*z + (mysize[0]+1)*y + x) + 2] = ((double)(z+start[2])) * cellSize[2];
			}
		}
	}
}

enum STATUS
{
	INSIDE,
	OUTSIDE,
	BORDER
};

struct Vertex
{
	double x,y,z;
};

static bool isContainedInSphere(const struct Vertex point, void const * const * const args)
{
	const struct Vertex *center = (struct Vertex *) args[0];
	const double *radius = (double *) args[1];
	const double diffX = point.x - center->x;
	const double diffY = point.y - center->y;
	const double diffZ = point.z - center->z;
	const double distance = sqrt(pow(diffX,2)+pow(diffY,2)+pow(diffZ,2));
	return distance < *radius;
}

static STATUS getCellStatus(const unsigned int cellNo, unsigned int *cells, double *vertices,
		bool (*isContainedInObject)(const struct Vertex point, void const * const * const args),
		void const * const * const args)
{
	STATUS lastStatus = BORDER;
	for(unsigned int i = 0; i < 8; i++) {
		const unsigned int vertexNo = cells[8*cellNo + i];
		const struct Vertex point = {vertices[3*vertexNo],vertices[3*vertexNo+1],vertices[3*vertexNo+2]};
		const STATUS currentStatus = isContainedInObject(point, args) ? INSIDE : OUTSIDE;
		if (lastStatus == BORDER) {
			lastStatus = currentStatus;
		} else {
			if (lastStatus == currentStatus) {
				continue;
			} else {
				return BORDER;
			}
		}
	}
	return lastStatus;
}

static STATUS getVertexStatus(const unsigned int vertexNo, double *vertices,
		bool (*isContainedInObject)(const struct Vertex point, void const * const * const args),
		void const * const * const args)
{
	const struct Vertex point = {vertices[3*vertexNo],vertices[3*vertexNo+1],vertices[3*vertexNo+2]};
	return isContainedInObject(point, args) ? INSIDE : OUTSIDE;
}

int main(int argc, char* argv[])
{
#ifdef USE_MPI
	MPI_Init(&argc, &argv);
#endif // USE_MPI

	utils::Args args;
#ifdef USE_HDF
	args.addOption("posix", 0, "use POSIX output (default: false)", utils::Args::No, false);
#endif // USE_HDF
	args.addOption("timesteps", 't', "the total number of time steps (default: 10)", utils::Args::Required, false);
	args.addOption("size", 's', "the inital number of cells in each dimension (default: 10)", utils::Args::Required, false);
	args.addOption("const", 'c', "the number of timesteps the mesh stays constant (default: 2)", utils::Args::Required, false);
	args.addOption("start", 0, "the timestep where benchmark should start (default: 0)", utils::Args::Required, false);
	args.addOption("no-vertex-filter", 'v', "disable the vertex filter", utils::Args::No, false);
	args.addOption("no-partition", 'p', "skip partition information", utils::Args::No, false);
	args.addAdditionalOption("filename", "the output file name");

	if (args.parse(argc, argv) != utils::Args::Success) {
#ifdef USE_MPI
		MPI_Finalize();
#endif // USE_MPI
		return 1;
	}

	int rank = 0;
	int nProcs = 1;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
#endif // USE_MPI

	std::vector<const char*> cellVariables;
	cellVariables.push_back("cell_sphere");
	cellVariables.push_back("cell_random");

	std::vector<const char*> vertexVariables;
	vertexVariables.push_back("vertex_sphere");
	vertexVariables.push_back("vertex_random");

	// Compute the number of ranks in each dimension
	int procs[3];

	procs[0] = pow(nProcs, 1./3.);
	while (nProcs % procs[0] != 0)
		procs[0]--;
	procs[1] = sqrt(nProcs / procs[0]);
	while ((nProcs / procs[0]) % procs[1] != 0)
		procs[1]--;
	procs[2] = nProcs / (procs[0] * procs[1]);
	logInfo(rank) << "Layout:" << procs[0] << 'x' << procs[1] << 'x' << procs[2];

	logInfo(rank) << "Initialize XDMF writer";

	const unsigned int start = args.getArgument<unsigned int>("start", 0);

	xdmfwriter::BackendType type = xdmfwriter::POSIX;
#ifdef USE_HDF
	if (!args.isSet("posix"))
		type = xdmfwriter::H5;
#endif // USE_HDF

	xdmfwriter::XdmfWriter<xdmfwriter::HEXAHEDRON, double> writer(
			type,
			args.getAdditionalArgument<const char*>("filename"),
			start);

	writer.init(cellVariables, vertexVariables,
		!args.isSet("no-vertex-filter"), !args.isSet("no-partition"));

	unsigned int numCells;
	unsigned int numVertices;
	unsigned int *cells = 0L;
	double *vertices = 0L;

	const unsigned int timesteps = args.getArgument<unsigned int>("timesteps", 10);
	const unsigned int initalSize = args.getArgument<unsigned int>("size", 10);
	const unsigned int constSteps = args.getArgument<unsigned int>("const", 2);

	unsigned int size[3] = {initalSize, initalSize, initalSize};
	unsigned int increaseDim = 0; // The next dimension we will increase

	// Compute the initial size
	for (unsigned int i = 0; i < start; i++) {
		if (i % constSteps == 0) {
			generateMesh(rank, procs, size, &numCells, &numVertices, cells, vertices);
			size[increaseDim++] *= 2;
			increaseDim = increaseDim % 3;
		}
	}

	if (start != 0) {
		// Set the mesh if we are restarting
		writer.setMesh(numCells, cells, numVertices, vertices, true);
	}

	for (unsigned int i = start; i < timesteps; i++) {
#ifdef USE_MPI
		// Synchronize to measure performance of writing a time step
		MPI_Barrier(MPI_COMM_WORLD);
#endif // USE_MPI

		logInfo(rank) << "Adding time step" << i << "to the output file";

		if (i % constSteps == 0) {
			generateMesh(rank, procs, size, &numCells, &numVertices, cells, vertices);
			size[increaseDim++] *= 2;
			increaseDim = increaseDim % 3;

			writer.setMesh(numCells, cells, numVertices, vertices);
		}

		double *cellData = new double[numCells];
		double *vertexData = new double[numVertices];
		writer.addTimeStep(i);

		const struct Vertex center = {0.5,0.5,0.5};
		const double radius = 0.5;
		const void * const containedInSphereArgs[2] = {&center, &radius};
		for (unsigned int k = 0; k < numCells; k++) {
			cellData[k] = (double)getCellStatus(k, cells, vertices, &isContainedInSphere, containedInSphereArgs);
		}
		writer.writeCellData(0, cellData);
		for (unsigned int k = 0; k < numCells; k++) {
			cellData[k] = rand();
		}
		writer.writeCellData(1, cellData);
		for (unsigned int k = 0; k < numVertices; k++) {
			vertexData[k] = (double)getVertexStatus(k, vertices, &isContainedInSphere, containedInSphereArgs);
		}
		writer.writeVertexData(0, vertexData);
		for (unsigned int k = 0; k < numVertices; k++) {
			vertexData[k] = rand();
		}
		writer.writeVertexData(1, vertexData);

		writer.flush();

		delete [] cellData;
		delete [] vertexData;
	}

	delete [] cells;
	delete [] vertices;

	writer.close();

#ifdef USE_MPI
	MPI_Finalize();
#endif // USE_MPI
	return 0;
}
