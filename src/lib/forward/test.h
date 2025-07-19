#pragma once
#include <stokes_pspg.h>

template <int dim>
class Test
{
public: 
    Test() = default;
    ~Test() = default;
    StokesPSPG<dim> stokes;

    void test();
};

template <int dim>
void Test<dim>::test()
{
    stokes.pcout << "this is a test" << std::endl;
}