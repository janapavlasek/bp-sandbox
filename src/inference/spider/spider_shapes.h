#ifndef BP_SANDBOX_INFERENCE_SPIDER_SPIDER_SHAPES_H
#define BP_SANDBOX_INFERENCE_SPIDER_SPIDER_SHAPES_H

#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <random>

#include <Eigen/Geometry>

#include "../utils/common_utils.h"
#include "observation.h"

#define EPS 1e-4
#define PER_PIX 0.1

namespace BPSandbox
{

namespace spider
{

typedef std::vector<std::vector<double> > ParticleList;
typedef std::map<std::string, std::vector<double> > ParticleState;
typedef std::map<std::string, ParticleList > ParticleStateList;

class Shape
{
public:
  Shape() :
    x(0),
    y(0),
    max_area(0)
  {};

  virtual bool pointInside(const double pt_x, const double pt_y) const = 0;
  virtual double likelihood(const Observation& obs) const = 0;
  virtual std::vector<double> get() const = 0;

  void set(std::vector<double> vals) { setPos(vals[0], vals[1]); };
  void setPos(double pos_x, double pos_y) { x = pos_x; y = pos_y; };

  double calcAverageVal(const Observation& obs, const double window, int& num_pts) const
  {
    double sum = 0;
    num_pts = 0;

    for (int i = static_cast<int>(x - window) - 1; i < static_cast<int>(x + window) + 2; ++i)
    {
      for (int j = static_cast<int>(y - window) - 1; j < static_cast<int>(y + window) + 2; ++j)
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

  double calcSDF(const Observation& obs, const double window) const
  {
    double sdf = 0;

    int start_x = std::max(0, static_cast<int>(std::floor(x - window)));
    int start_y = std::max(0, static_cast<int>(std::floor(y - window)));
    int end_x = std::min(static_cast<int>(obs.width), static_cast<int>(std::ceil(x + window)));
    int end_y = std::min(static_cast<int>(obs.height), static_cast<int>(std::ceil(y + window)));

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

  double x, y;
  double max_area;
};

class Circle : public Shape
{
public:
  Circle() :
    radius(10),
    radius_bounds({5, 14})
  {
    max_area = PI * radius_bounds[1] * radius_bounds[1];
  }

  Circle(const double pos_x, const double pos_y, const double r) :
    radius_bounds({5, 14})
  {
    set({pos_x, pos_y, r});
    max_area = PI * radius_bounds[1] * radius_bounds[1];
  }

  void set(std::vector<double> vals)
  {
    setPos(vals[0], vals[1]);
    radius = vals[2];
    radius = std::max(radius_bounds[0], radius);
    radius = std::min(radius_bounds[1], radius);
  }

  std::vector<double> get() const
  {
    std::vector<double> v({x, y, radius});
    return v;
  }

  double radius;
  std::vector<double> radius_bounds;

  double averageVal(const Observation& obs, int& num_pts) const
  {
    return calcAverageVal(obs, radius, num_pts);
  }

  double sdf(const Observation& obs) const
  {
    return calcSDF(obs, radius);
  }

  double likelihood(const Observation& obs) const
  {
    return sdf(obs);
  }

  bool pointInside(const double pt_x, const double pt_y) const
  {
    return pow(pt_x - x, 2) + pow(pt_y - y, 2) <= radius * radius;
  }
};

class Rectangle : public Shape
{
public:
  Rectangle() :
    width(27),
    height(8),
    theta(0),
    width_bounds({12, 42}),
    height_bounds({2, 15})
  {
    max_area = 500;
  }

  Rectangle(const double pos_x, const double pos_y, const double theta, const double w, const double h) :
    width_bounds({12, 42}),
    height_bounds({2, 15})
  {
    set({pos_x, pos_y, theta, w, h});

    max_area = 500;  // width_bounds[1] * height_bounds[1];
  }

  void set(std::vector<double> vals)
  {
    setPos(vals[0], vals[1]);
    theta = vals[2];
    width = std::max(width_bounds[0], vals[3]);
    width = std::min(width_bounds[1], vals[3]);
    height = std::max(height_bounds[0], vals[4]);
    height = std::min(height_bounds[1], vals[4]);
  }

  std::vector<double> get() const
  {
    std::vector<double> v({x, y, theta, width, height});
    return v;
  }

  double width, height, theta;
  std::vector<std::vector<double> > corner_pts;
  std::vector<double> width_bounds, height_bounds;

  double averageVal(const Observation& obs, int& num_pts) const
  {
    return calcAverageVal(obs, width, num_pts);
  }

  double sdf(const Observation& obs) const
  {
    double sub_size = std::max(width, height);
    return calcSDF(obs, sub_size);
  }

  double likelihood(const Observation& obs) const
  {
    return sdf(obs);
  }

  void setPoints(const std::vector<std::vector<double> >& pts)
  {
    corner_pts = pts;
  }

  bool pointInside(const double pt_x, const double pt_y) const
  {
    // Algorithm to check if a point is inside a polygon.
    //   (https://en.wikipedia.org/wiki/Point_in_polygon)
    int num_intersect = 0;

    std::vector<double> origin({0, 0});
    std::vector<double> pt({pt_x, pt_y});

    for (size_t i = 0; i < 4; ++i)
    {
      if (intersect(origin, pt, corner_pts[i], corner_pts[(i + 1) % 4])) num_intersect++;
    }

    return num_intersect % 2 != 0;
  }

private:
  bool ccw(const std::vector<double>& A,
           const std::vector<double>& B,
           const std::vector<double>& C) const
  {
    return (C[1]-A[1])*(B[0]-A[0]) >= (B[1]-A[1])*(C[0]-A[0]);
  }

  bool intersect(const std::vector<double>& A,
                 const std::vector<double>& B,
                 const std::vector<double>& C,
                 const std::vector<double>& D) const
  {
    // Algorithm to check intersection of two line segments.
    //   (https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/)
    return ccw(A,C,D) != ccw(B,C,D) && ccw(A,B,C) != ccw(A,B,D);
  }
};

};  // namespace spider
};  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_SPIDER_SPIDER_SHAPES_H
