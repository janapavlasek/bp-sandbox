#ifndef BP_SANDBOX_INFERENCE_COMMON_SPIDER_GRAPH_H
#define BP_SANDBOX_INFERENCE_COMMON_SPIDER_GRAPH_H

#include <memory>

#include "../utils/mixture_model.h"
#include "spider_pairwise.h"
#include "spider_shapes.h"

namespace BPSandbox
{

namespace spider
{

class Node
{
public:
  Node() :
    id_(0)
  {};

  Node(const int id, std::shared_ptr<Shape> shape) :
    id_(id),
    shape_(shape)
  {};

  int id() const { return id_; };
  void setObs(const Observation& obs) { obs_ = obs; }
  void setBelief(const utils::MixtureModel& belief) { belief_ = belief; }

  void setBelief(const std::vector<std::vector<double> > means, const std::vector<double> cov)
  {
    std::vector<utils::GaussianMultiRV> gaussians;
    std::vector<double> weights;

    for (auto& mean : means)
    {
      gaussians.push_back(utils::GaussianMultiRV(mean, cov));
      weights.push_back(1);
    }
    belief_ = utils::MixtureModel(gaussians, weights);
  }

  ParticleList means() const
  {
    ParticleList mu;
    for (auto& comp : belief_.components())
    {
      mu.push_back(comp.mean());
    }
    return mu;
  }

  double unary(const std::vector<double> x)
  {
    shape_->set(x);
    return shape_->likelihood(obs_);
  }

private:
  int id_;
  Observation obs_;
  std::shared_ptr<Shape> shape_;
  utils::MixtureModel belief_;

};

class Edge
{
public:
  Edge() {};
  Edge(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2,
       std::shared_ptr<PairwisePotential> psi12, std::shared_ptr<PairwisePotential> psi21) :
    n1_(n1),
    n2_(n2),
    psi12_(psi12),
    psi21_(psi21)
  {};

  bool in(const Node& n) const { return n.id() == n1_->id() || n.id() == n2_->id(); }

  utils::MixtureModel msgFrom(const Node& from)
  {
    if (!in(from)) return utils::MixtureModel();

    if (from.id() == n1_->id())
    {
      return m12_;  // Message from n1 to n2.
    }
    else if (from.id() == n2_->id())
    {
      return m21_;  // Message from n2 to n1.
    }
  }

  utils::MixtureModel msgTo(const Node& to)
  {
    if (!in(to)) return utils::MixtureModel();

    if (to.id() == n1_->id())
    {
      return m21_;  // Message from n2 to n1.
    }
    else if (to.id() == n2_->id())
    {
      return m12_;  // Message from n1 to n2.
    }
  }

  void updateMessage(const Node& from, const Node& to, const utils::MixtureModel& msg)
  {
    if (from.id() == n1_->id() && to.id() == n2_->id())
    {
      m12_ = msg;  // Message from n1 to n2.
    }
    else if (from.id() == n2_->id() && to.id() == n1_->id())
    {
      m21_ = msg;  // Message from n2 to n1.
    }
    else
    {
      std::cerr << "Incorrect nodes for this edge." << std::endl;
    }
  }

  std::shared_ptr<Shape> sample(std::shared_ptr<Shape> x, const Node& to)
  {
    if (!in(to)) return nullptr;

    if (to.id() == n1_->id()) return psi21_->pairwiseSample(x);
    else if (to.id() == n2_->id()) return psi12_->pairwiseSample(x);
  }

private:

  std::shared_ptr<Node> n1_, n2_;
  utils::MixtureModel m12_, m21_;
  std::shared_ptr<PairwisePotential> psi12_, psi21_;
};


class SpiderGraph
{
public:
  SpiderGraph() :
    num_rects_(8),
    circle_cov_({2, 2, 2}),
    rect_cov_({2, 2, 0.2, 2, 2})
  {
    // Root node.
    nodes_.push_back(std::make_shared<Node>(0, std::make_shared<Circle>()));

    // Add all rectangles.
    for (size_t i = 0; i < num_rects_; ++i)
    {
      nodes_.push_back(std::make_shared<Node>(i + 1, std::make_shared<Rectangle>()));
    }

    // Connect the inner links to the root.
    for (size_t i = 1; i <= num_rects_ / 2; ++i)
    {
      edges_.push_back(Edge(nodes_[0], nodes_[i],
                            std::make_shared<LinkToRoot>(), std::make_shared<RootToLink>(i - 1)));
    }

    // Connect the inner links to the outer links.
    for (size_t i = 1; i <= num_rects_ / 2; ++i)
    {
      edges_.push_back(Edge(nodes_[i], nodes_[i + 4],
                            std::make_shared<InnerLinkToOuterLink>(), std::make_shared<OuterLinkToInnerLink>()));
    }
  };

  void setNodeBelief(const ParticleList& means, const int id)
  {
    if (id == 0)
    {
      nodes_[id]->setBelief(means, circle_cov_);
    }
    else
    {
      nodes_[id]->setBelief(means, circle_cov_);
    }
  }

  size_t num_rects() { return num_rects_; }

  void setObs(const Observation& obs)
  {
    for (auto& n : nodes_)
    {
      n->setObs(obs);
    }
  }

  std::vector<utils::MixtureModel> getNeighbourMsgs(const Node& n)
  {
    std::vector<utils::MixtureModel> msgs;
    for (auto& e: edges_)
    {
      if (!e.in(n)) continue;
      msgs.push_back(e.msgTo(n));
    }

    return msgs;
  }

  std::vector<utils::MixtureModel> getNeighbourMsgs(const Node& n, const Node& except)
  {
    std::vector<utils::MixtureModel> msgs;
    for (auto& e: edges_)
    {
      if (!e.in(n)) continue;
      if (e.in(except)) continue;  // Skip the edge to this node.
      msgs.push_back(e.msgTo(n));
    }

    return msgs;
  }

  ParticleStateList toStateMap() const
  {
    ParticleStateList particle_map;

    ParticleList circle_means;
    for (auto& mean : nodes_[0]->means())
    {
      circle_means.push_back(mean);
    }
    particle_map.insert({"circles", circle_means});

    for (size_t i = 0; i < num_rects_; ++i)
    {
      ParticleList rect_means;
      for (auto& mean : nodes_[i + 1]->means())
      {
        rect_means.push_back(mean);
      }
      particle_map.insert({"l" + std::to_string(i + 1), rect_means});
    }

    return particle_map;
  }

private:
  std::vector<Edge> edges_;
  std::vector<std::shared_ptr<Node> > nodes_;

  std::vector<double> circle_cov_, rect_cov_;
  size_t num_rects_;

};

}  // namespace spider
}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_SPIDER_GRAPH_H
