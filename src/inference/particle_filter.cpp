#include "common/inference_utils.h"
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
  update_count_ = 0;
  particles_.clear();
  weights_.clear();

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> pix_dist(0, obs_.width - 1);
  std::normal_distribution<float> h_dist{8, 2};
  std::normal_distribution<float> w_dist{27, 5};
  std::normal_distribution<float> r_dist{10, 2};
  std::normal_distribution<float> dist{0, PI / 8};

  for (size_t i = 0; i < num_particles; ++i)
  {
    std::vector<float> joints;
    for (size_t i = 0; i < num_joints_ / 2; ++i)
    {
      joints.push_back(normalize_angle(i * PI / 2 + dist(gen)));
    }
    for (size_t i = num_joints_ / 2; i < num_joints_; ++i)
    {
      joints.push_back(dist(gen));
    }

    SpiderParticle sp(pix_dist(gen), pix_dist(gen), r_dist(gen), w_dist(gen), h_dist(gen), joints);
    particles_.push_back(sp);
  }

  weights_ = reweight(particles_, obs_);

  return particlesToMap(particles_);
}

std::map<std::string, ParticleList> ParticleFilter::update()
{
  // Add noise to particles, but keep the best one.
  auto best = particleEstimate();
  particles_ = jitterParticles(particles_, 2, 0.1, 2);
  particles_.push_back(best);

  weights_ = reweight(particles_, obs_);
  particles_ = resample(particles_, weights_);

  update_count_++;

  return particlesToMap(particles_);
}

std::vector<double> ParticleFilter::reweight(const std::vector<SpiderParticle>& particles, const Observation& obs)
{
  std::vector<double> weights;
  for (auto& p: particles)
  {
    double w = p.jointUnaryLikelihood(obs);
    weights.push_back(w);
  }

  return weights;
}

std::vector<SpiderParticle> ParticleFilter::resample(const std::vector<SpiderParticle>& particles, std::vector<double>& weights)
{
  std::vector<double> normalized_weights = normalizeVector(weights, true);
  // std::vector<size_t> keep = importanceSample(num_particles_, normalized_weights);
  std::vector<size_t> keep = lowVarianceSample(num_particles_, normalized_weights);

  std::vector<SpiderParticle> new_particles;
  std::vector<double> new_weights;

  for (auto& idx : keep)
  {
    new_particles.push_back(particles[idx]);
    new_weights.push_back(weights[idx]);
  }

  weights = new_weights;

  return new_particles;
}

std::map<std::string, ParticleList> ParticleFilter::estimate()
{
  std::vector<SpiderParticle> est({particleEstimate()});

  return particlesToMap(est);
}

SpiderParticle ParticleFilter::particleEstimate()
{
  if (particles_.size() != weights_.size())
  {
    std::cerr << "PANIC! Can't get estimate. " << particles_.size() << " != " << weights_.size() << std::endl;
  }
  size_t best = 0;
  for (size_t i = 1; i < weights_.size(); ++i)
  {
    if (weights_[i] > weights_[best])
    {
      best = i;
    }
  }

  return particles_[best];
}

std::map<std::string, ParticleList> ParticleFilter::particlesToMap(const std::vector<SpiderParticle>& particles)
{
  std::map<std::string, ParticleList> particle_map;

  for (auto& n : link_names_)
  {
    particle_map.insert({n, ParticleList()});
  }

  for (auto& p : particles)
  {
    for (auto& data : p.toPartStates())
    {
      particle_map[data.first].push_back(data.second);
    }
  }
  return particle_map;
}


}  // namespace BPSandbox
