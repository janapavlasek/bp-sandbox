#include "particle_filter.h"

namespace BPSandbox
{

ParticleFilter::ParticleFilter() :
  num_joints_(8)
{
  link_names_.push_back("circles");
  for (size_t i = 0; i < num_joints_; ++i)
  {
    link_names_.push_back("l" + std::to_string(i + 1));
  }
}

std::map<std::string, ParticleList> ParticleFilter::init(const int num_particles)
{
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
  return particlesToMap();
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
