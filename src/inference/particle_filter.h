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

typedef std::map<std::string, std::vector<float> > ParticleState;

class SpiderParticle
{
public:
  SpiderParticle(int x, int y, std::vector<float> joints) :
    x(x),
    y(y),
    joints(joints),
    num_joints(8),
    link_length(26.6),
    link_spacing(26.6)
  {
  }

  // Graph attributes.
  size_t num_joints;
  float link_length;
  float link_spacing;

  // Graph state.
  int x, y;
  std::vector<float> joints;

  ParticleState toPartStates()
  {
    ParticleState state;
    std::vector<float> root_pos({x, y});
    state.insert({"circles", root_pos});

    // This is the first layer of joints, connected to the root.
    for (size_t i = 0; i < num_joints / 2; ++i)
    {
      std::string name = "l" + std::to_string(i + 1);
      float center_x = x + (link_length / 2 + link_spacing) * cos(joints[i]);
      float center_y = y + (link_length / 2 + link_spacing) * sin(joints[i]);
      float theta = RAD_TO_DEG * joints[i];

      std::vector<float> p({center_x, center_y, theta});
      state.insert({name, p});
    }

    // This is the second layer of joints, each connected to one in the first layer.
    for (size_t i = num_joints / 2; i < num_joints; ++i)
    {
      float parent_joint = joints[i - num_joints / 2];
      Eigen::Translation<float, 2> t1(link_length + link_spacing, 0);
      Eigen::Rotation2D<float> rot1(parent_joint);
      Eigen::Translation<float, 2> t2(link_length / 2 + link_spacing, 0);
      Eigen::Rotation2D<float> rot2(joints[i]);
      Eigen::Translation<float, 2> tw(x, y);

      Eigen::Vector2f pt(0, 0);
      auto new_pt = tw * rot1 * t1 * rot2 * t2 * pt;

      std::string name = "l" + std::to_string(i + 1);

      float theta = RAD_TO_DEG * normalize_angle(joints[i] + parent_joint);

      std::vector<float> p({new_pt[0], new_pt[1], theta});
      state.insert({name, p});
    }

    return state;
  }
};

class ParticleFilter
{
public:
  ParticleFilter();

  std::map<std::string, ParticleList> init(const int num_particles);
  std::map<std::string, ParticleList> update();

private:

  std::map<std::string, ParticleList> particlesToMap();

  Observation obs_;
  std::vector<SpiderParticle> particles_;
  std::vector<float> weights_;

  size_t num_joints_;
  std::vector<std::string> link_names_;
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_PARTICLE_FILTER_H
