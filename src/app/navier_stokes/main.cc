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

#include <navier_stokes_gls.h>

using Utilities::MPI::MPI_InitFinalize;

int main(int argc, char* argv[])
{
  try
  {
    MPI_InitFinalize mpi_init(argc, argv, 1);

    RuntimeParams_NavierStokes params;
    params.read_params("params.prm");

#if SPATIAL_DIMENSION == 2
    NavierStokesGLS<2> navier(params);
    navier.run();
    
#elif SPATIAL_DIMENSION == 3
    NavierStokesGLS<3> navier(params);
    navier.run();
    
#else
    static_assert(
      SPATIAL_DIMENSION == 2 ||
      SPATIAL_DIMENSION  == 3,
      "Unsupported SPATIAL_DIMENSION");
#endif
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