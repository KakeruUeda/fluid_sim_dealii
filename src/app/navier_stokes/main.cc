/* ---------------------
 * navier_stokes.cc
 *
 * Copyright (C) 2025
 * All rights reserved.
 *
 * ---------------------
 *
 * Author: Kakeru Ueda
 * Date: Jun, 2025
 */

#include <navier_stokes.h>

using Utilities::MPI::MPI_InitFinalize;

int main(int argc, char* argv[])
{
  try
  {
    MPI_InitFinalize mpi_init(argc, argv, 1);

    RuntimeParams_NavierStokes params;
    params.read_params("params.prm");

    NavierStokesGLS<3> navier(params);
    navier.run();
  }
  catch (std::exception& exc)
  {
    std::cerr << "Exeption on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting .." << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "--------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "--------------------------"
              << std::endl;
    return 1;
  }
}