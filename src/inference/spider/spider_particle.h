#ifndef BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H
#define BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H

#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <random>

#include <Eigen/Geometry>

#include "../utils/common_utils.h"
#include "observation.h"
#include "spider_shapes.h"

namespace BPSandbox
{

namespace spider
{

class SpiderParticle
{
public:
  SpiderParticle(const double x, const double y,
                 const double r, const double w, const double h,
                 const std::vector<double>& joints) :
    x(x),
    y(y),
    w(w),
    h(h),
    joints(joints),
    num_joints(8),
    root(Circle(x, y, r))
  {
    double min = 4.0;
    root.radius = std::max(r, min);
    updateLinks(std::max(w, min), std::max(h, min));
  }

  // Graph attributes.
  size_t num_joints;

  Circle root;
  std::vector<Rectangle> links;

  // Graph state.
  double x, y;
  double w, h;
  std::vector<double> joints;

  void updateLinks(const double w, const double h)
  {
    links.clear();

    for (size_t i = 0; i < num_joints; ++i)
    {
      Eigen::Transform<float,2,Eigen::Affine> rect_tf;
      Eigen::Translation<float, 2> tw(x, y);
      Eigen::Translation<float, 2> rect_center_tf(w / 2 + w, 0);
      float theta;

      if (i < num_joints / 2)
      {
        // This is the first layer of joints, connected to the root.
        theta = joints[i];
        Eigen::Translation<float, 2> t1(w / 2 + w, 0);
        Eigen::Rotation2D<float> rot1(theta);
        rect_tf = tw * rot1;
      }
      else
      {
        // This is the second layer of joints, connected to the first layer.
        float parent_joint = joints[i - num_joints / 2];
        theta = utils::normalize_angle(joints[i] + parent_joint);
        Eigen::Translation<float, 2> t1(w + w, 0);
        Eigen::Rotation2D<float> rot1(parent_joint);
        Eigen::Rotation2D<float> rot2(joints[i]);
        rect_tf = tw * rot1 * t1 * rot2;
      }

      Eigen::Vector2f pt(0, 0);
      auto new_pt = rect_tf * rect_center_tf * pt;

      Rectangle r(new_pt[0], new_pt[1], theta, w, h);

      // Get four corners.
      Eigen::Translation<float, 2> top_left_tf(w, r.height / 2);
      Eigen::Translation<float, 2> bottom_left_tf(w, -r.height / 2);
      Eigen::Translation<float, 2> top_right_tf(w + w, r.height / 2);
      Eigen::Translation<float, 2> bottom_right_tf(w + w, -r.height / 2);

      auto top_left = rect_tf * top_left_tf * pt;
      auto bottom_left = rect_tf * bottom_left_tf * pt;
      auto top_right = rect_tf * top_right_tf * pt;
      auto bottom_right = rect_tf * bottom_right_tf * pt;

      std::vector<std::vector<double> > rect_pts({{top_left[0], top_left[1]},
                                                  {top_right[0], top_right[1]},
                                                  {bottom_right[0], bottom_right[1]},
                                                  {bottom_left[0], bottom_left[1]}});

      r.setPoints(rect_pts);

      links.push_back(r);
    }
  }

  ParticleState toPartStates() const
  {
    ParticleState state;
    std::vector<double> root_pos({x, y, root.radius});
    state.insert({"circles", root_pos});

    // Get all the rectangles.
    for (size_t i = 0; i < links.size(); ++i)
    {
      std::string name = "l" + std::to_string(i + 1);

      std::vector<double> p({links[i].x, links[i].y, links[i].theta, links[i].width, links[i].height});
      state.insert({name, p});
    }

    return state;
  }

  bool pointInside(const double pt_x, const double pt_y) const
  {
    if (root.pointInside(pt_x, pt_y)) return true;

    for (auto& l : links)
    {
      if (l.pointInside(pt_x, pt_y)) return true;
    }
    return false;
  }

  bool inBounds(const double x, const double y) const
  {
    if (root.x < 0 || root.x >= x || root.y < 0 || root.y >= y) return false;

    for (auto& l : links)
    {
      if (l.x < 0 || l.x >= x || l.y < 0 || l.y >= y) return false;
    }
    return true;
  }

  double iou(const Observation& obs) const
  {
    // Intersection over union in a small square.
    int sub_size = w * 4;  // Half the size of the sub observation.
    float intersect = 0;
    float uni = 0;

    int start_x = std::max(0, static_cast<int>(std::floor(x - sub_size)));
    int start_y = std::max(0, static_cast<int>(std::floor(y - sub_size)));
    int end_x = std::min(static_cast<int>(obs.width), static_cast<int>(std::ceil(x + sub_size)));
    int end_y = std::min(static_cast<int>(obs.height), static_cast<int>(std::ceil(y + sub_size)));

    for (size_t i = start_x; i < end_x; ++i)
    {
      for (size_t j = start_y; j < end_y; ++j)
      {
        bool point_inside = pointInside(i, j);
        if (point_inside && obs.getPixel(i, j) == 1) intersect++;
        if (point_inside || obs.getPixel(i, j) == 1) uni++;
      }
    }

    return intersect / uni;
  }

  double sdf(const Observation& obs) const
  {
    double sdf = log(root.sdf(obs));

    for (auto& l : links)
    {
      sdf += log(l.sdf(obs));
    }

    return sdf;  // std::max(0.0, sdf);
  }

  double jointUnaryLikelihood(const Observation& obs) const
  {
    return sdf(obs);
  }

  void print() const
  {
    std::cout << "x: " << x << ", y: " << y;
    std::cout << ", r: " << root.radius << ", w: " << links[0].width << ", h: " << links[0].height;
    std::cout << std::endl;
  }

  void save(const int w, const int h, const std::string file_name) const
  {
    // TODO.
    for (size_t i = 0; i < w; ++i)
    {
      for (size_t j = 0; j < h; ++j)
      {
        continue;
      }
    }
  }
};

typedef std::vector<SpiderParticle> SpiderList;

inline ParticleStateList particlesToMap(const SpiderList& particles)
{
  std::map<std::string, ParticleList> particle_map;

  if (particles.size() < 1) return particle_map;

  particle_map.insert({"circles", ParticleList()});
  for (size_t i = 0; i < particles[0].num_joints; ++i)
  {
    particle_map.insert({"l" + std::to_string(i + 1), ParticleList()});
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

}  // namespace spider
}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H
