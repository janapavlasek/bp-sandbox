#ifndef BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H
#define BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H

#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <random>

#include "common_utils.h"
#include "spider_particle.h"

namespace BPSandbox
{

template <class T>
static std::vector<double> normalizeVector(const std::vector<T>& vals, const bool log_likelihood = true)
{
  std::vector<double> normalized_vals;
  if (vals.size() < 1) return normalized_vals;

  // Make sure vals are positive.
  auto min_w = *std::min_element(vals.begin(), vals.end());
  assert(log_likelihood || min_w >= 0);

  double sum = 0;

  for (auto& w : vals) {
    if (log_likelihood)
    {
      sum += exp(w - min_w);
    }
    else
    {
      sum += w;
    }
  }

  for (auto& w : vals) {
    if (sum == 0) {
      normalized_vals.push_back(1.0 / vals.size());
    } else {
      double new_w;
      if (log_likelihood) new_w = exp(w - min_w) / sum;
      else                new_w = w / sum;
      normalized_vals.push_back(new_w);
    }
  }

  return normalized_vals;
}

static std::vector<size_t> importanceSample(const size_t num_particles,
                                            const std::vector<double>& normalized_weights,
                                            const bool keep_best = true)
{
  std::vector<size_t> sample_ind;

  if (num_particles < 1 || normalized_weights.size() < 1) return sample_ind;

  if (keep_best)
  {
    size_t max_idx = 0;
    for (size_t i = 1; i < normalized_weights.size(); ++i)
    {
      if (normalized_weights[i] > normalized_weights[max_idx])
      {
        max_idx = i;
      }
    }
    sample_ind.push_back(max_idx);
  }

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  while (sample_ind.size() < num_particles)
  {
    float r = distribution(gen);
    int idx = 0;
    float sum = normalized_weights[idx];
    while (sum < r) {
      ++idx;
      sum += normalized_weights[idx];
    }
    sample_ind.push_back(idx);
  }

  return sample_ind;
}

static std::vector<size_t> lowVarianceSample(const size_t num_particles,
                                             const std::vector<double>& normalized_weights)
{
  std::vector<size_t> sample_ind;

  if (num_particles < 1 || normalized_weights.size() < 1) return sample_ind;

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<float> distribution(0.0, 1.0 / num_particles);
  float r = distribution(gen);
  int idx = 0;
  float sum = normalized_weights[idx];

  for (size_t i = 0; i < num_particles; ++i)
  {
    float u = r + i * (1. / num_particles);
    while (u > sum)
    {
      idx++;
      sum += normalized_weights[idx];
    }
    sample_ind.push_back(idx);
  }

  return sample_ind;
}

static spider::SpiderParticle jitterParticle(const spider::SpiderParticle& particle, const float jitter_pix,
                                             const float jitter_angle, const float jitter_param)
{
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> dpix{0, jitter_pix};
  std::normal_distribution<float> dangle{0, jitter_angle};
  std::normal_distribution<float> dparam{0, jitter_param};

  std::vector<float> new_joints;
  for (auto& j : particle.joints)
  {
    new_joints.push_back(j + dangle(gen));
  }

  spider::SpiderParticle new_particle(particle.x + dpix(gen), particle.y + dpix(gen),
                                      particle.root.radius + dparam(gen),
                                      particle.links[0].width + dparam(gen),
                                      particle.links[0].height + dparam(gen),
                                      new_joints);

  return new_particle;
}

static spider::SpiderList jitterParticles(const spider::SpiderList& particles,
                                          const float jitter_pix, const float jitter_angle,
                                          const float jitter_param)
{
  spider::SpiderList new_particles;

  for (auto& p : particles)
  {
    new_particles.push_back(jitterParticle(p, jitter_pix, jitter_angle, jitter_param));
  }

  return new_particles;
}

};  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H
