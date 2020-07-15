#ifndef BP_SANDBOX_INFERENCE_COMMON_UTILS_H
#define BP_SANDBOX_INFERENCE_COMMON_UTILS_H

#define PI 3.14159265
#define RAD_TO_DEG 180 / PI

#include <cmath>

namespace BPSandbox
{

static inline float normalize_angle(float angle)
{
  const double result = fmod(angle, 2.0*M_PI);
  if(result < 0) return result + 2.0*M_PI;
  return result;
}

static inline double sigmoid(double x, double alpha = 0.1)
{
  return 2 / (exp(-alpha * x) + 1) - 1;
}

};  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_UTILS_H
