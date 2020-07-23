#ifndef BP_SANDBOX_INFERENCE_COMMON_SPIDER_PAIRWISE_H
#define BP_SANDBOX_INFERENCE_COMMON_SPIDER_PAIRWISE_H

#include <math.h>
#include <memory>

#include "spider_particle.h"

namespace BPSandbox
{

namespace spider
{

class PairwisePotential
{
public:
  PairwisePotential() :
    std_p(10),
    std_s(2),
    std_alpha(0.26)  // 15 degrees
  {};
  virtual double pairwise(std::shared_ptr<Shape>& n1, std::shared_ptr<Shape>& n2) = 0;
  virtual std::shared_ptr<Shape> pairwiseSample(std::shared_ptr<Shape>& n) = 0;

  float noise(float std)
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0, std};

    return dist(gen);
  }

  float pNoise() { return noise(std_p); }
  float sNoise() { return noise(std_s); }
  float alphaNoise() { return noise(std_alpha); }

  float std_p, std_s, std_alpha;
};

class RootToLink : public PairwisePotential
{
public:
  RootToLink(const int i) :
    delta_w(28. / 5.),
    delta_h(4. / 5.),
    C(7),
    joint_idx(i)
  {};

  double pairwise(std::shared_ptr<Shape>& n1, std::shared_ptr<Shape>& n2)
  {
    return 0;
  }

  std::shared_ptr<Shape> pairwiseSample(std::shared_ptr<Shape>& n)
  {
    auto vals = n->get();  // Circle with {x, y, r}.
    double x = vals[0] + pNoise();
    double y = vals[1] + pNoise();
    double theta = joint_idx * M_PI / 2 + alphaNoise();
    double w = 2 * C * vals[2] * delta_w * delta_h / (C * delta_h + delta_w) + sNoise();
    double h = 2 * vals[2] * delta_w * delta_h / (C * delta_h + delta_w) + sNoise();

    std::shared_ptr<Rectangle> r = std::make_shared<Rectangle>(x, y, theta, w, h);
    return r;
  }

  float delta_w, delta_h, C;
  int joint_idx;
};

class LinkToRoot : public PairwisePotential
{
public:
  LinkToRoot() :
    delta_w(28. / 5.),
    delta_h(4. / 5.)
  {};

  double pairwise(std::shared_ptr<Shape>& n1, std::shared_ptr<Shape>& n2)
  {
    return 0;
  }

  std::shared_ptr<Shape> pairwiseSample(std::shared_ptr<Shape>& n)
  {
    auto vals = n->get();  // Rectangle with {x, y, theta, width, height}.
    double x = vals[0] + pNoise();
    double y = vals[1] + pNoise();
    double r = 0.5 * (vals[3] / delta_w + vals[4] / delta_h) + sNoise();

    std::shared_ptr<Circle> c = std::make_shared<Circle>(x, y, r);
    return c;
  }

  float delta_w, delta_h;
};

class InnerLinkToOuterLink : public PairwisePotential
{
public:
  InnerLinkToOuterLink() {};

  double pairwise(std::shared_ptr<Shape>& n1, std::shared_ptr<Shape>& n2)
  {
    return 0;
  }

  std::shared_ptr<Shape> pairwiseSample(std::shared_ptr<Shape>& n)
  {
    auto vals = n->get();  // Rectangle with {x, y, theta, width, height}.
    // Given the inner link, generate an outer link.
    double x_j = vals[0] + vals[3] * cos(vals[2]) + pNoise();
    double y_j = vals[1] + vals[3] * sin(vals[2]) + pNoise();
    double theta_j = vals[2] + alphaNoise();
    double w_j = vals[3] + sNoise();
    double h_j = vals[4] + sNoise();

    std::shared_ptr<Rectangle> r = std::make_shared<Rectangle>(x_j, y_j, theta_j, w_j, h_j);
    return r;
  }
};

class OuterLinkToInnerLink : public PairwisePotential
{
public:
  OuterLinkToInnerLink() {};

  double pairwise(std::shared_ptr<Shape>& n1, std::shared_ptr<Shape>& n2)
  {
    return 0;
  }

  std::shared_ptr<Shape> pairwiseSample(std::shared_ptr<Shape>& n)
  {
    auto vals = n->get();  // Rectangle with {x, y, theta, width, height}.
    // Given the outer node, generate an inner node.
    double x_i = vals[0] + pNoise();
    double y_i = vals[1] + pNoise();
    x_i += vals[3] * cos(vals[2] + M_PI) + pNoise();
    y_i += vals[3] * sin(vals[2] + M_PI) + pNoise();
    double theta_i = vals[2] + alphaNoise();

    double w_i = vals[3] + sNoise();
    double h_i = vals[4] + sNoise();

    std::shared_ptr<Rectangle> r = std::make_shared<Rectangle>(x_i, y_i, theta_i, w_i, h_i);
    return r;
  }
};

}  // namespace spider
}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_SPIDER_PAIRWISE_H
