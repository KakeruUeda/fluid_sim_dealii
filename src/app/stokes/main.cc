#include <stokes_problem.h>
using MY_MPI = Utilities::MPI::MPI_InitFinalize;

int main(int argc, char* argv[])
{
  try
  {
    MY_MPI mpi_init(argc, argv, 1);

    StokesPSPG<3> stokes;
    stokes.run();

  }
  catch (std::exception& exc)
  {
    std::cerr << "Exeption on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting .." << std::endl;
    return 1;
  }
}