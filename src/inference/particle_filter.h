#ifndef BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H
#define BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H

#include <cmath>
#include <string>
#include <vector>
#include <random>

#include "common/observation.h"
#include "common/spider_particle.h"

namespace BPSandbox
{

class ParticleFilter
{
public:
  ParticleFilter();

  spider::ParticleStateList init(const int num_particles, const bool use_obs = true);
  spider::ParticleStateList update();
  spider::ParticleStateList estimate();

private:
  spider::SpiderParticle particleEstimate();
  spider::SpiderParticle randomParticle(const float x, const float y, const float r);
  std::vector<double> reweight(const spider::SpiderList& particles, const Observation& obs);
  spider::SpiderList resample(const spider::SpiderList& particles, std::vector<double>& weights);

  size_t num_particles_;
  size_t update_count_;
  size_t num_joints_;

  Observation obs_;
  spider::SpiderList particles_;
  std::vector<double> weights_;
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H
