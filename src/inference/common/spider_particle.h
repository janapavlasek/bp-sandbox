#ifndef BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H
#define BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H

#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <random>

#include <Eigen/Geometry>

#include "common_utils.h"
#include "observation.h"

#define EPS 1e-4
#define PER_PIX 0.1

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
    y(0),
    radius_bounds({5, 14})
  {
    max_area = PI * radius_bounds[1] * radius_bounds[1];
  }

  Circle(const float x, const float y, const float r) :
    radius(r),
    x(x),
    y(y),
    radius_bounds({5, 14})
  {
    radius = std::max(radius_bounds[0], radius);
    radius = std::min(radius_bounds[1], radius);
    max_area = PI * radius_bounds[1] * radius_bounds[1];
  }

  float radius;
  float x, y;
  float max_area;
  std::vector<float> radius_bounds;

  double calcAverageVal(const Observation& obs, int& num_pts) const
  {
    double sum = 0;
    num_pts = 0;
    for (int i = static_cast<int>(x - radius) - 1; i < static_cast<int>(x + radius) + 2; ++i)
    {
      for (int j = static_cast<int>(y - radius) - 1; j < static_cast<int>(y + radius) + 2; ++j)
      {
        if (pointInside(i, j))
        {
          num_pts++;
          sum += obs.getPixel(i, j);
        }
      }
    }

    return sum;
  }

  double sdf(const Observation& obs) const
  {
    double sdf = 0;

    int start_x = std::max(0, static_cast<int>(std::floor(x - radius)));
    int start_y = std::max(0, static_cast<int>(std::floor(y - radius)));
    int end_x = std::min(static_cast<int>(obs.width), static_cast<int>(std::ceil(x + radius)));
    int end_y = std::min(static_cast<int>(obs.height), static_cast<int>(std::ceil(y + radius)));

    for (int i = start_x; i < end_x; ++i)
    {
      for (int j = start_y; j < end_y; ++j)
      {
        bool point_inside = pointInside(i, j);
        if (point_inside)
        {
          if (obs.isOccupied(i, j))
          {
            sdf += PER_PIX;
          }
          else
          {
            sdf -= PER_PIX;
          }
        }
      }
    }

    sdf = sdf / (PER_PIX * max_area);

    return std::max(EPS, sdf);
  }

  bool pointInside(const float pt_x, const float pt_y) const
  {
    return pow(pt_x - x, 2) + pow(pt_y - y, 2) <= radius * radius;
  }
};

class Rectangle
{
public:
  Rectangle() :
    width(27),
    height(8),
    x(0),
    y(0),
    theta(0),
    width_bounds({12, 42}),
    height_bounds({2, 15})
  {
  }

  Rectangle(const float x, const float y, const float theta, const float w, const float h) :
    width(w),
    height(h),
    x(x),
    y(y),
    theta(theta),
    width_bounds({12, 42}),
    height_bounds({2, 15})
  {
    width = std::max(width_bounds[0], width);
    width = std::min(width_bounds[1], width);
    height = std::max(height_bounds[0], height);
    height = std::min(height_bounds[1], height);

    max_area = 500;  // width_bounds[1] * height_bounds[1];
  }

  float width, height;
  float x, y, theta;
  float max_area;
  std::vector<std::vector<float> > corner_pts;
  std::vector<float> width_bounds, height_bounds;

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
          sum += obs.getPixel(i, j);
        }
      }
    }

    return sum;
  }

  double sdf(const Observation& obs) const
  {
    double sdf = 0;

    float sub_size = std::max(width, height);

    int start_x = std::max(0, static_cast<int>(std::floor(x - sub_size)));
    int start_y = std::max(0, static_cast<int>(std::floor(y - sub_size)));
    int end_x = std::min(static_cast<int>(obs.width), static_cast<int>(std::ceil(x + sub_size)));
    int end_y = std::min(static_cast<int>(obs.height), static_cast<int>(std::ceil(y + sub_size)));

    for (int i = start_x; i < end_x; ++i)
    {
      for (int j = start_y; j < end_y; ++j)
      {
        bool point_inside = pointInside(i, j);
        if (point_inside)
        {
          if (obs.isOccupied(i, j))
          {
            sdf += PER_PIX;
          }
          else
          {
            sdf -= PER_PIX;
          }
        }
      }

    }

    sdf = sdf / (PER_PIX * max_area);

    return std::max(EPS, sdf);
  }

  void setPoints(const std::vector<std::vector<float> >& pts)
  {
    corner_pts = pts;
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

private:
  bool ccw(const std::vector<float>& A,
           const std::vector<float>& B,
           const std::vector<float>& C) const
  {
    return (C[1]-A[1])*(B[0]-A[0]) >= (B[1]-A[1])*(C[0]-A[0]);
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
};

class SpiderParticle
{
public:
  SpiderParticle(const float x, const float y,
                 const float r, const float w, const float h,
                 const std::vector<float>& joints) :
    x(x),
    y(y),
    w(w),
    h(h),
    joints(joints),
    num_joints(8),
    root(Circle(x, y, r))
  {
    float min = 4.0;
    root.radius = std::max(r, min);
    updateLinks(std::max(w, min), std::max(h, min));
  }

  // Graph attributes.
  size_t num_joints;

  Circle root;
  std::vector<Rectangle> links;

  // Graph state.
  float x, y;
  float w, h;
  std::vector<float> joints;

  void updateLinks(const float w, const float h)
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
        theta = normalize_angle(joints[i] + parent_joint);
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
    std::vector<float> root_pos({x, y, root.radius});
    state.insert({"circles", root_pos});

    // Get all the rectangles.
    for (size_t i = 0; i < links.size(); ++i)
    {
      std::string name = "l" + std::to_string(i + 1);

      std::vector<float> p({links[i].x, links[i].y, links[i].theta, links[i].width, links[i].height});
      state.insert({name, p});
    }

    return state;
  }

  bool pointInside(const float pt_x, const float pt_y) const
  {
    if (root.pointInside(pt_x, pt_y)) return true;

    for (auto& l : links)
    {
      if (l.pointInside(pt_x, pt_y)) return true;
    }
    return false;
  }

  bool inBounds(const float x, const float y) const
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

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_SPIDER_PARTICLE_H
