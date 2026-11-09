/******************************************************************************
Test driver: the IPM executable (src/main_ipm.cu, same options and outputs) with
the perimeter deposit disabled, so that every triangle takes the direct deposit
as when the step buffers do not fit. Built by tests/gpu/build_rung.sh as
ipm_gpu_direct (-D is_float) and ipm_gpu_double_direct (-D is_double); the
hook it sets is not reachable from the shipped executables or the Python API.
******************************************************************************/

#define main ipm_main
#include "../../src/main_ipm.cu"
#undef main

int main(int argc, char* argv[])
{
	IPM<dtype>::testing_force_direct_deposit = true;
	return ipm_main(argc, argv);
}
