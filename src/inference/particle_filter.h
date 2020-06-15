#ifndef BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H
#define BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <Eigen/Geometry>

#include "inference_utils.h"

namespace BPSandbox
{

class ParticleFilter
{
public:
  ParticleFilter();

  std::map<std::string, ParticleList> init(const int num_particles);
  std::map<std::string, ParticleList> update();
  std::map<std::string, ParticleList> estimate();

private:

  SpiderParticle particleEstimate();
  std::map<std::string, ParticleList> particlesToMap(const std::vector<SpiderParticle>& particles);
  std::vector<double> reweight(const std::vector<SpiderParticle>& particles, const Observation& obs);
  std::vector<SpiderParticle> resample(const std::vector<SpiderParticle>& particles, std::vector<double>& weights);

  size_t num_particles_;
  size_t update_count_;

  Observation obs_;
  std::vector<SpiderParticle> particles_;
  std::vector<double> weights_;

  size_t num_joints_;
  std::vector<std::string> link_names_;
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H
