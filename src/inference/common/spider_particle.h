#ifndef BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H
#define BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H

#include <cmath>
#include <iostream>
#include <algorithm>
#include <map>
#include <random>

#include <Eigen/Geometry>

#include "common_utils.h"
#include "observation.h"

namespace BPSandbox
{

typedef std::vector<std::vector<float> > ParticleList;
typedef std::map<std::string, std::vector<float> > ParticleState;
typedef std::map<std::string, ParticleList > ParticleStateList;

class Circle
{
public:
  Circle() :
    radius(10),
    x(0),
    y(0)
  {
  }

  Circle(const float x, const float y) :
    radius(10),
    x(x),
    y(y)
  {
  }

  float radius;
  float x, y;

  double calcAverageVal(const Observation& obs, int& num_pts) const
  {
    double sum = 0;
    num_pts = 0;
    for (int i = static_cast<int>(x - radius) - 1; i < static_cast<int>(x + radius) + 2; ++i)
    {
      for (int j = static_cast<int>(y - radius) - 1; j < static_cast<int>(y + radius) + 2; ++j)
      {
        if (pow(i - x, 2) + pow(j - y, 2) <= radius * radius)
        {
          num_pts++;
          sum += (1 - obs.getPixel(i, j));
        }
      }
    }

    return sum;
  }
};

class Rectangle
{
public:
  Rectangle() :
    width(26.6),
    height(7.6),
    x(0),
    y(0),
    theta(0)
  {
  }

  Rectangle(const float x, const float y, const float theta) :
    width(26.6),
    height(7.6),
    x(x),
    y(y),
    theta(theta)
  {
  }

  float width, height;
  float x, y, theta;
  std::vector<std::vector<float> > corner_pts;

  double calcAverageVal(const Observation& obs, int& num_pts) const
  {
    double sum = 0;
    num_pts = 0;

    for (int i = static_cast<int>(x - width) - 1; i < static_cast<int>(x + width) + 2; ++i)
    {
      for (int j = static_cast<int>(y - width) - 1; j < static_cast<int>(y + width) + 2; ++j)
      {
        if (pointInside(i, j))
        {
          num_pts++;
          sum += (1 - obs.getPixel(i, j));
        }
      }
    }

    return sum;
  }

  void setPoints(const std::vector<std::vector<float> >& pts)
  {
    corner_pts = pts;
  }

private:
  bool ccw(const std::vector<float>& A,
           const std::vector<float>& B,
           const std::vector<float>& C) const
  {
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0]);
  }

  bool intersect(const std::vector<float>& A,
                 const std::vector<float>& B,
                 const std::vector<float>& C,
                 const std::vector<float>& D) const
  {
    // Algorithm to check intersection of two line segments.
    //   (https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/)
    return ccw(A,C,D) != ccw(B,C,D) && ccw(A,B,C) != ccw(A,B,D);
  }

  bool pointInside(const float pt_x, const float pt_y) const
  {
    // Algorithm to check if a point is inside a polygon.
    //   (https://en.wikipedia.org/wiki/Point_in_polygon)
    int num_intersect = 0;

    std::vector<float> origin({0, 0});
    std::vector<float> pt({pt_x, pt_y});

    for (size_t i = 0; i < 4; ++i)
    {
      if (intersect(origin, pt, corner_pts[i], corner_pts[(i + 1) % 4])) num_intersect++;
    }

    return num_intersect % 2 != 0;
  }
};

class SpiderParticle
{
public:
  SpiderParticle(const float x, const float y, const std::vector<float>& joints) :
    x(x),
    y(y),
    joints(joints),
    num_joints(8),
    link_length(26.6),
    link_spacing(26.6)
  {
    root.x = x;
    root.y = y;
    updateLinks();
  }

  // Graph attributes.
  size_t num_joints;
  float link_length;
  float link_spacing;

  Circle root;
  std::vector<Rectangle> links;

  // Graph state.
  float x, y;
  std::vector<float> joints;

  void updateLinks()
  {
    links.clear();

    for (size_t i = 0; i < num_joints; ++i)
    {
      Eigen::Transform<float,2,Eigen::Affine> rect_tf;
      Eigen::Translation<float, 2> tw(x, y);
      Eigen::Translation<float, 2> rect_center_tf;
      float theta;

      if (i < num_joints / 2)
      {
        // This is the first layer of joints, connected to the root.
        theta = joints[i];
        Eigen::Translation<float, 2> t1(link_length / 2 + link_spacing, 0);
        Eigen::Rotation2D<float> rot1(theta);
        rect_center_tf = Eigen::Translation<float, 2>(link_length / 2 + link_spacing, 0);
        rect_tf = tw * rot1;
      }
      else
      {
        // This is the second layer of joints, connected to the first layer.
        float parent_joint = joints[i - num_joints / 2];
        theta = normalize_angle(joints[i] + parent_joint);
        Eigen::Translation<float, 2> t1(link_length + link_spacing, 0);
        Eigen::Rotation2D<float> rot1(parent_joint);
        Eigen::Rotation2D<float> rot2(joints[i]);
        rect_center_tf = Eigen::Translation<float, 2>(link_length / 2 + link_spacing, 0);
        rect_tf = tw * rot1 * t1 * rot2;
      }

      Eigen::Vector2f pt(0, 0);
      auto new_pt = rect_tf * rect_center_tf * pt;

      Rectangle r(new_pt[0], new_pt[1], theta);

      // Get four corners.
      Eigen::Translation<float, 2> top_left_tf(link_spacing, r.height / 2);
      Eigen::Translation<float, 2> bottom_left_tf(link_spacing, -r.height / 2);
      Eigen::Translation<float, 2> top_right_tf(link_spacing + link_length, r.height / 2);
      Eigen::Translation<float, 2> bottom_right_tf(link_spacing + link_length, -r.height / 2);

      auto top_left = rect_tf * top_left_tf * pt;
      auto bottom_left = rect_tf * bottom_left_tf * pt;
      auto top_right = rect_tf * top_right_tf * pt;
      auto bottom_right = rect_tf * bottom_right_tf * pt;

      std::vector<std::vector<float> > rect_pts({{top_left[0], top_left[1]},
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
    std::vector<float> root_pos({x, y});
    state.insert({"circles", root_pos});

    // Get all the rectangles.
    for (size_t i = 0; i < links.size(); ++i)
    {
      std::string name = "l" + std::to_string(i + 1);

      std::vector<float> p({links[i].x, links[i].y, links[i].theta});
      state.insert({name, p});
    }

    return state;
  }

  double jointUnaryLikelihood(const Observation& obs) const
  {
    double w = 0;
    int num_pts = 0;
    int total_pts = 0;

    // This is the unary for the whole particle so we mark the whole grid as unchecked.
    obs.markUnchecked();

    // Circle first.
    w += root.calcAverageVal(obs, num_pts);
    total_pts += num_pts;

    for (size_t i = 0; i < links.size(); i++)
    {
      w += links[i].calcAverageVal(obs, num_pts);
      total_pts += num_pts;
    }

    return w / total_pts;
  }
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H
