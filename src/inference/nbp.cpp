#include "nbp.h"

namespace BPSandbox
{

NBP::NBP() :
  num_particles_(100),
  update_count_(0)
{
}

spider::ParticleStateList NBP::init(const int num_particles, const bool use_obs)
{
  num_particles_ = num_particles;
  update_count_ = 0;

  auto obs_circ = obs_.getCircles();
  auto obs_rect = obs_.getRectangles();

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> pix_dist(0, obs_.width - 1);
  std::uniform_int_distribution<int> idx_dist(0, obs_circ.size() - 1);

  // Initialize circle belief.
  spider::ParticleList circle_means;
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

    circle_means.push_back(utils::jitter({x, y, r}, {10, 10, 2}));
  }

  graph_.setNodeBelief(circle_means, 0);

  // Initialize rectangle beliefs.
  idx_dist = std::uniform_int_distribution<int>(0, obs_rect.size() - 1);
  for (size_t i = 0; i < graph_.num_rects(); ++i)
  {
    spider::ParticleList rect_means;
    for (size_t i = 0; i < num_particles; ++i)
    {
      float x, y, theta = 0, w = 27, h = 8;
      if (use_obs)
      {
        auto rect_sample = obs_rect[idx_dist(gen)];
        x = rect_sample[1];
        y = rect_sample[0];
        theta = rect_sample[2];
        w = rect_sample[3];
        h = rect_sample[4];
      }
      else
      {
        x = pix_dist(gen);
        y = pix_dist(gen);
      }

      rect_means.push_back(utils::jitter({x, y, theta, w, h}, {10, 10, 0.2, 2, 2}));
    }

    graph_.setNodeBelief(rect_means, i + 1);
  }

  return graph_.toStateMap();
}

spider::ParticleStateList NBP::update()
{
  spider::ParticleStateList p;
  update_count_++;
  return p;
}

spider::ParticleStateList NBP::estimate()
{
  spider::ParticleStateList p;
  return p;
}

}  // namespace BPSandbox
