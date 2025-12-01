#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "cuda_runtime.h"

namespace opensn
{
void
CBC_AngleSet::CreateCUDAStream()
{
  cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(&stream_ptr);
  cudaError_t err = cudaStreamCreate(stream);
}

void
CBC_AngleSet::DestroyCUDAStream()
{
  cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(&stream_ptr);
  cudaError_t err = cudaStreamDestroy(*stream);
  stream_ptr = nullptr;
}
}  // namespace opensn