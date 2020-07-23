#ifndef BP_SANDBOX_INFERENCE_NBP_H
#define BP_SANDBOX_INFERENCE_NBP_H

#include <cmath>
#include <string>
#include <vector>
#include <random>

#include "spider/observation.h"
#include "spider/spider_graph.h"

namespace BPSandbox
{

class NBP
{
public:
  NBP();

  spider::ParticleStateList init(const int num_particles, const bool use_obs = true);
  spider::ParticleStateList update();
  spider::ParticleStateList estimate();

private:
  size_t num_particles_;
  size_t update_count_;

  Observation obs_;
  spider::SpiderGraph graph_;
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_NBP_H
