#include "common/inference_utils.h"
#include "particle_filter.h"

namespace BPSandbox
{

ParticleFilter::ParticleFilter() :
  num_joints_(8),
  num_particles_(50),
  update_count_(0)
{
}

spider::ParticleStateList ParticleFilter::init(const int num_particles, const bool use_obs)
{
  num_particles_ = num_particles;
  update_count_ = 0;

  particles_.clear();
  weights_.clear();

  auto obs_circ = obs_.getCircles();

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> pix_dist(0, obs_.width - 1);
  std::uniform_int_distribution<int> idx_dist(0, obs_circ.size() - 1);

  for (size_t i = 0; i < num_particles; ++i)
  {
    float x, y, r = 10;
    if (use_obs)
    {
      auto circ_sample = obs_circ[idx_dist(gen)];
      x = circ_sample[1];
      y = circ_sample[0];
      r = circ_sample[2];
    }
    else
    {
      x = pix_dist(gen);
      y = pix_dist(gen);
    }

    particles_.push_back(randomParticle(x, y, r));
  }

  weights_ = reweight(particles_, obs_);

  return spider::particlesToMap(particles_);
}

spider::SpiderParticle ParticleFilter::randomParticle(const float x, const float y, const float r)
{
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> pix_dist(0, 10);
  std::normal_distribution<float> h_dist{8, 2};
  std::normal_distribution<float> w_dist{27, 5};
  std::normal_distribution<float> r_dist{0, 2};
  std::normal_distribution<float> theta_dist{0, PI / 8};

  std::vector<float> joints;
  for (size_t i = 0; i < num_joints_ / 2; ++i)
  {
    joints.push_back(normalize_angle(i * PI / 2 + theta_dist(gen)));
  }
  for (size_t i = num_joints_ / 2; i < num_joints_; ++i)
  {
    joints.push_back(theta_dist(gen));
  }

  spider::SpiderParticle sp(x + pix_dist(gen), y + pix_dist(gen), r + r_dist(gen),
                            w_dist(gen), h_dist(gen), joints);

  return sp;
}

spider::ParticleStateList ParticleFilter::update()
{
  // Add noise to particles, but keep the best one.
  auto best = particleEstimate();
  particles_ = jitterParticles(particles_, 2, 0.1, 2);
  particles_.push_back(best);

  weights_ = reweight(particles_, obs_);
  particles_ = resample(particles_, weights_);

  update_count_++;

  return spider::particlesToMap(particles_);
}

std::vector<double> ParticleFilter::reweight(const spider::SpiderList& particles, const Observation& obs)
{
  std::vector<double> weights;
  for (auto& p: particles)
  {
    double w = p.jointUnaryLikelihood(obs);
    weights.push_back(w);
  }

  return weights;
}

spider::SpiderList ParticleFilter::resample(const spider::SpiderList& particles, std::vector<double>& weights)
{
  std::vector<double> normalized_weights = normalizeVector(weights, true);
  // std::vector<size_t> keep = importanceSample(num_particles_, normalized_weights);
  std::vector<size_t> keep = lowVarianceSample(num_particles_, normalized_weights);

  spider::SpiderList new_particles;
  std::vector<double> new_weights;

  for (auto& idx : keep)
  {
    new_particles.push_back(particles[idx]);
    new_weights.push_back(weights[idx]);
  }

  weights = new_weights;

  return new_particles;
}

spider::ParticleStateList ParticleFilter::estimate()
{
  spider::SpiderList est({particleEstimate()});
  return spider::particlesToMap(est);
}

spider::SpiderParticle ParticleFilter::particleEstimate()
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

}  // namespace BPSandbox
