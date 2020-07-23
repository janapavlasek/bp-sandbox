#ifndef BP_SANDBOX_INFERENCE_COMMON_UTILS_H
#define BP_SANDBOX_INFERENCE_COMMON_UTILS_H

#define PI 3.14159265
#define RAD_TO_DEG 180 / PI

#include <cmath>
#include <random>

namespace BPSandbox
{

namespace utils
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

static inline std::vector<double> jitter(const std::vector<double>& x, const std::vector<double>& params)
{
  if (x.size() != params.size()) return x;

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::vector<double> noisy;
  for (size_t i = 0; i < x.size(); ++i)
  {
    std::normal_distribution<double> dist{0, params[i]};
    noisy.push_back(x[i] + dist(gen));
  }

  return noisy;
}

};  // namespace utils
};  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_UTILS_H
