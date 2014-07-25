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

#include <cxxtest/GlobalFixture.h>

// Workaround to use MPI in CxxTests
int cxxtest_main(int, char**);

static bool mpiInitSuccess = true;

int main(int argc, char** argv)
{
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
		mpiInitSuccess = false;

	int exitCode = cxxtest_main(argc, argv);

	MPI_Finalize();

	return exitCode;
}

/**
 * Init and finalize MPI (requires the main functions)
 *
 * @see main
 */
class MPIFixture : public CxxTest::GlobalFixture
{
public:
	bool setUpWorld()
	{
		return mpiInitSuccess;
	}
};
static MPIFixture mpiFixture;
