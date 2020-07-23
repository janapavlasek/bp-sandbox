#ifndef BP_SANDBOX_INFERENCE_COMMON_MIXTURE_MODEL_H
#define BP_SANDBOX_INFERENCE_COMMON_MIXTURE_MODEL_H

#include <math.h>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "inference_utils.h"

namespace BPSandbox
{
namespace utils
{

class GaussianRV
{
public:
  GaussianRV(const double mean, const double std):
    mean_(mean),
    std_(std)
  {};

  double mean() const { return mean_; }
  double std() const { return std_; }

  double pdf(const double x) const
  {
    return exp(-0.5 * pow((x - mean_) / std_, 2)) / (std_ * sqrt(2 * M_PI));
  }

  double sample() const
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> dist{mean_, std_};

    return dist(gen);
  }

private:
  double mean_, std_;

};


class GaussianMultiRV
{
public:
  GaussianMultiRV() :
    dim_(0)
  {};

  GaussianMultiRV(const std::vector<double>& mean,
                  const std::vector<double>& cov) :
    mean_(mean),
    cov_(cov),
    dim_(mean.size())
  {};

  std::vector<double> mean() const { return mean_; }
  std::vector<double> cov() const { return cov_; }
  size_t dim() const { return dim_; }

  double pdf(const std::vector<double>& val) const
  {
    std::vector<double> in_val(val);
    Eigen::Map<Eigen::VectorXd> x(in_val.data(), dim_);
    Eigen::Map<Eigen::VectorXd> mu(mean().data(), dim_);
    Eigen::MatrixXd sigma = Eigen::Map<Eigen::VectorXd>(cov().data(), dim_).asDiagonal();

    double y = pow(2 * M_PI, -dim_ / 2.0) * pow(sigma.determinant(), -0.5);
    auto tmp = (x - mu).transpose() * sigma.inverse() * (x - mu);
    y *= exp(-0.5 * tmp(0));
    return y;
  }

  std::vector<double> sample() const
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> dist{0, 1};

    Eigen::VectorXd z(dim_);
    for (size_t i = 0; i < dim_; ++i) z << dist(gen);

    Eigen::Map<Eigen::MatrixXd> sigma(cov().data(), dim_, dim_);
    Eigen::Map<Eigen::VectorXd> mu(mean().data(), dim_);

    Eigen::LLT<Eigen::MatrixXd> llt(sigma); // compute the Cholesky decomposition of A
    Eigen::MatrixXd L = llt.matrixL();

    Eigen::VectorXd x = mu + L * z;
    std::vector<double> s(x.data(), x.data() + x.size());

    return s;
  }

private:
  std::vector<double> mean_, cov_;
  size_t dim_;

};


class MixtureModel
{
public:
  MixtureModel() :
    num_components_(0)
  {}

  MixtureModel(const std::vector<GaussianMultiRV>& gaussians,
               const std::vector<double>& weights) :
    gaussians_(gaussians),
    weights_(weights)
  {
    num_components_ = gaussians_.size();
    normalize();
  };

  void normalize() { weights_ = normalizeVector(weights_, false); }

  // Getters.
  GaussianMultiRV at(const size_t idx) const { return gaussians_[idx]; }

  double weight(const size_t idx) const { return weights_[idx]; }

  size_t num_components() const { return num_components_; }

  std::vector<double> weights() const { return weights_; }

  std::vector<GaussianMultiRV> components() const { return gaussians_; }

  void add(const GaussianMultiRV& gaussian, double w)
  {
    gaussians_.push_back(gaussian);
    weights_.push_back(w);
    num_components_++;
  }

  void add(const MixtureModel& mix)
  {
    auto new_comp = mix.components();
    auto new_w = mix.weights();
    gaussians_.insert(gaussians_.end(), new_comp.begin(), new_comp.end());
    weights_.insert(weights_.begin(), new_w.begin(), new_w.end());

    num_components_ = gaussians_.size();
  }

  double pdf(const std::vector<double>& x) const
  {
    double y = 0;
    for (size_t i = 0; i < num_components_; ++i)
    {
      y += weights_[i] * gaussians_[i].pdf(x);
    }
    return y;
  }

  MixtureModel product(const MixtureModel& dist) const
  {
    std::vector<GaussianMultiRV> new_gaussians;
    std::vector<double> new_weights;

    for (size_t i = 0; i < num_components_; ++i)
    {
      for (size_t j = 0; j < dist.num_components(); ++j)
      {
        Eigen::MatrixXd cov_self = Eigen::Map<Eigen::VectorXd>(at(i).cov().data(), at(i).dim()).asDiagonal();
        Eigen::MatrixXd cov_dist = Eigen::Map<Eigen::VectorXd>(dist.at(j).cov().data(), dist.at(j).dim()).asDiagonal();

        Eigen::MatrixXd sigma = (cov_self.inverse() + cov_dist.inverse()).inverse();

        Eigen::Map<Eigen::VectorXd> mu_self(at(i).mean().data(), at(i).dim());
        Eigen::Map<Eigen::VectorXd> mu_dist(dist.at(j).mean().data(), dist.at(j).dim());
        Eigen::VectorXd mu = sigma * (cov_self.inverse() * mu_self + cov_dist.inverse() * mu_dist);

        Eigen::VectorXd sig_data = sigma.diagonal();
        std::vector<double> sigma_v(sig_data.data(), sig_data.data() + sig_data.size());
        std::vector<double> mu_v(mu.data(), mu.data() + mu.size());
        GaussianMultiRV g(mu_v, sigma_v);

        double w = weight(i) * at(i).pdf(mu_v) * dist.weight(j) * dist.at(j).pdf(mu_v) / g.pdf(mu_v);

        new_gaussians.push_back(g);
        new_weights.push_back(w);
      }
    }

    return MixtureModel(new_gaussians, new_weights);
  }

private:

  size_t num_components_;
  std::vector<GaussianMultiRV> gaussians_;
  std::vector<double> weights_;

};

inline GaussianMultiRV gaussianProduct(const std::vector<GaussianMultiRV>& gaussians)
{
  if (gaussians.size() < 1) return GaussianMultiRV();

  // Assume that the covariance is the same for all the components.
  auto old_cov = gaussians[0].cov();
  Eigen::MatrixXd cov_i = Eigen::Map<Eigen::VectorXd>(old_cov.data(), old_cov.size()).asDiagonal();

  Eigen::VectorXd cov(gaussians[0].dim());
  double scale = 1. / gaussians.size();
  for (size_t i = 0; i < cov.size(); ++i) cov << scale * old_cov[i];

  Eigen::VectorXd mu = Eigen::VectorXd::Zero(cov.size());
  for (size_t i = 0; i < gaussians.size(); ++i)
  {
    Eigen::Map<Eigen::VectorXd> mu_i(gaussians[i].mean().data(), gaussians[i].dim());
    mu += mu_i;
  }
  mu = cov.asDiagonal() * cov_i.inverse() * mu;

  std::vector<double> cov_v(cov.data(), cov.data() + cov.size());
  std::vector<double> mu_v(mu.data(), mu.data() + mu.size());

  return GaussianMultiRV(mu_v, cov_v);
}

/**
 * Uses Gibbs sampling to sample one value from a product of mixtures.
 * @param  mixtures Mixture models, length d.
 * @param  M        Number of components per model.
 * @return          A sample from the mixture product.
 */
inline std::vector<double> gibbsSampleOneFromProduct(const std::vector<MixtureModel>& mixtures,
                                                     const size_t M, const size_t k,
                                                     double (*f)(std::vector<double>),
                                                     double& w)
{
  size_t d = mixtures.size();
  std::vector<size_t> labels(d, 0);

  // Initialize the component labels, based on the weights.
  for (size_t j = 0; j < d; ++j)
  {
    labels[j] = importanceSample(1, mixtures[j].weights())[0];
  }

  for (size_t iter = 0; iter < k; ++iter)
  {
    for (size_t j = 0; j < d; ++j)
    {
      // Calculate the product of the components other than j.
      std::vector<GaussianMultiRV> prod_star;
      for (size_t mix = 0; mix < d; ++mix)
      {
        if (mix != j) prod_star.push_back(mixtures[mix].at(labels[mix]));
      }
      auto f_star = gaussianProduct(prod_star);

      // Calculate the weight for each component.
      std::vector<double> wj;
      for (size_t i = 0; i < M; ++i)
      {
        auto comp_i = mixtures[j].at(i);
        auto f_i = gaussianProduct({f_star, comp_i});
        auto mu_i = f_i.mean();
        double w_i = mixtures[j].weight(i) * comp_i.pdf(mu_i) * f_star.pdf(mu_i) / f_i.pdf(mu_i);
        w_i *= (*f)(mu_i);

        wj.push_back(w_i);
      }
      wj = normalizeVector(wj, false);
      labels[k] = importanceSample(1, wj)[0];
    }
  }

  std::vector<GaussianMultiRV> prod;
  for (size_t j = 0; j < d; ++j)
  {
    prod.push_back(mixtures[j].at(labels[j]));
  }

  auto g = gaussianProduct(prod);
  auto x = g.sample();

  w = (*f)(x) / (*f)(g.mean());
  return x;
}

/**
 * Uses Gibbs sampling to approximate the product of d mixtures.
 * @param  mixtures Mixture models, length d.
 * @param  M        Number of components per model.
 * @return          The approximated product of mixtures with M components.
 */
inline MixtureModel gibbsProduct(const std::vector<MixtureModel>& mixtures,
                                  const size_t M, const size_t k,
                                  double (*f)(std::vector<double>))
{
  MixtureModel mm;

  for (size_t i = 0; i < M; ++i)
  {
    double w;
    std::vector<double> mean = gibbsSampleOneFromProduct(mixtures, M, k, f, w);
    std::vector<double> cov(mean.size(), 0.1);

    mm.add(GaussianMultiRV(mean, cov), w);
  }

  mm.normalize();

  return mm;
}

};  // namespace BPSandbox
}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_MIXTURE_MODEL_H
