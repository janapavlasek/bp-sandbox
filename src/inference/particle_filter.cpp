#include "particle_filter.h"

namespace BPSandbox
{

ParticleFilter::ParticleFilter() :
  num_joints_(8),
  num_particles_(50),
  update_count_(0)
{
  link_names_.push_back("circles");
  for (size_t i = 0; i < num_joints_; ++i)
  {
    link_names_.push_back("l" + std::to_string(i + 1));
  }
}

std::map<std::string, ParticleList> ParticleFilter::init(const int num_particles)
{
  num_particles_ = num_particles;
  particles_.clear();
  weights_.clear();

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<int> int_dist(0, obs_.width - 1);
  std::uniform_real_distribution<float> dist(0, 2 * PI);

  for (size_t i = 0; i < num_particles; ++i)
  {
    std::vector<float> joints;
    for (size_t i = 0; i < num_joints_; ++i)
    {
      joints.push_back(dist(gen));
    }

    SpiderParticle sp(int_dist(gen), int_dist(gen), joints);
    particles_.push_back(sp);
  }

  return particlesToMap();
}

std::map<std::string, ParticleList> ParticleFilter::update()
{
  // Add noise to particles.
  particles_ = jitterParticles(particles_, 3, 0.1);
  weights_ = reweight(particles_, obs_);
  particles_ = resample(particles_, weights_);

  update_count_++;

  return particlesToMap();
}

std::vector<double> ParticleFilter::reweight(const std::vector<SpiderParticle>& particles, const Observation& obs)
{
  std::vector<double> weights;
  for (auto& p: particles)
  {
    double w = p.averageUnaryLikelihood(obs);
    weights.push_back(w);
  }

  return weights;
}

std::vector<SpiderParticle> ParticleFilter::resample(const std::vector<SpiderParticle>& particles, const std::vector<double>& weights)
{
  std::vector<double> normalized_weights = normalizeVector(weights, false);
  std::vector<size_t> keep = importanceSample(num_particles_, normalized_weights);

  std::vector<SpiderParticle> new_particles;

  for (auto& idx : keep)
  {
    new_particles.push_back(particles[idx]);
  }

  return new_particles;
}

std::map<std::string, ParticleList> ParticleFilter::particlesToMap()
{
  std::map<std::string, ParticleList> particle_map;

  for (auto& n : link_names_)
  {
    particle_map.insert({n, ParticleList()});
  }

  for (auto& p : particles_)
  {
    for (auto& data : p.toPartStates())
    {
      particle_map[data.first].push_back(data.second);
    }
  }
  return particle_map;
}


}  // namespace BPSandbox
