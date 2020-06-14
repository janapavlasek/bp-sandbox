#ifndef BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H
#define BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H

#define PI 3.14159265
#define RAD_TO_DEG 180 / PI

#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

namespace BPSandbox
{

static inline float normalize_angle(float angle)
{
  const double result = fmod(angle, 2.0*M_PI);
  if(result < 0) return result + 2.0*M_PI;
  return result;
}

typedef std::vector<std::vector<float> > ParticleList;
typedef std::map<std::string, std::vector<float> > ParticleState;
typedef std::map<std::string, ParticleList > ParticleStateList;

class Observation
{
public:
  Observation() :
    file_path_("/home/jana/code/bp-sandbox/media/obs.pbm")
  {
    loadImage(file_path_);
  }

  float getPixel(const int i, const int j) const
  {
    // col, row
    if (i < 0 || i >= width || j < 0 || j >= height)
    {
      return 0;
    }
    return data_[i * width + j];
  }

  size_t width, height;

private:

  void loadImage(const std::string& file_path)
  {
    std::ifstream fin(file_path);
    std::string line;

    if(!fin)
    {
      std::cerr << "Error reading observation file " << file_path << std::endl;
      return;
    }

    // The first line can be ignored.
    if (!std::getline(fin, line)) return;
    // The secong line has width and height.
    if (std::getline(fin, line))
    {
      std::stringstream ss(line);
      ss >> width >> height;
    }
    else return;

    data_ = new float[width * height];

    // Get the pixels.
    for (int row = 0; row < height; ++row)
    {
      if (!std::getline(fin, line)) break;
      std::stringstream ss(line);
      for (int col = 0; col < width; ++col)
      {
        std::string pix;
        ss >> pix;
        data_[row * width + col] = std::stof(pix);
      }
    }
  }

  float* data_;
  std::string file_path_;
};

/*
 * Shape and graph info.
 */
class Circle
{
public:
  Circle() :
    radius(20),
    x(0),
    y(0)
  {
  }

  Circle(const float x, const float y) :
    radius(20),
    x(x),
    y(y)
  {
  }

  double calcAverageVal(const Observation& obs) const
  {
    double sum = 0;
    int count = 0;
    for (int i = static_cast<int>(x - radius) - 1; i < static_cast<int>(x + radius) + 2; ++i)
    {
      for (int j = static_cast<int>(y - radius) - 1; j < static_cast<int>(y + radius) + 2; ++j)
      {
        if (pow(i - x, 2) + pow(j - y, 2) <= radius * radius)
        {
          count++;
          sum += obs.getPixel(i, j);
        }
      }
    }
    if (count == 0) return 0;
    return sum / count;
  }

  float radius;
  float x, y;
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

  double calcAverageVal(const Observation& obs) const
  {
    double sum = 0;
    int count = 0;

    // Line parameters.
    float a = tan(theta);
    float a_perp = tan(normalize_angle(theta + PI / 2));
    float b = y - a * x;
    float b_perp = y - a_perp * x;

    // Line offsets.
    float x0_h = (height / 2) / cos(theta);
    float x0_w = (width / 2) / cos(theta);

    for (int i = static_cast<int>(x - width) - 1; i < static_cast<int>(x + width) + 2; ++i)
    {
      for (int j = static_cast<int>(y - width) - 1; j < static_cast<int>(y + width) + 2; ++j)
      {
        if (j >= a * (i + x0_h) + b && j <= a * (i - x0_h) + b &&
            j >= a_perp * (i - x0_w) + b_perp && j <= a_perp * (i + x0_w) + b_perp)
        {
          count++;
          sum += obs.getPixel(i, j);
        }
      }
    }

    if (count == 0) return 0;
    return sum / count;
  }

  float width, height;
  float x, y, theta;
};

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
  int x, y;
  std::vector<float> joints;

  void updateLinks()
  {
    links.clear();

    // This is the first layer of joints, connected to the root.
    for (size_t i = 0; i < num_joints / 2; ++i)
    {
      float center_x = x + (link_length / 2 + link_spacing) * cos(joints[i]);
      float center_y = y + (link_length / 2 + link_spacing) * sin(joints[i]);
      float theta = joints[i];

      Rectangle r(center_x, center_y, theta);
      links.push_back(r);
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

      float theta = normalize_angle(joints[i] + parent_joint);

      Rectangle r(new_pt[0], new_pt[1], theta);
      links.push_back(r);
    }
  }

  ParticleState toPartStates()
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

  double averageUnaryLikelihood(const Observation obs) const
  {
    double w = 0;
    // Circle first.
    w += root.calcAverageVal(obs);

    for (size_t i = 0; i < links.size(); i++)
    {
      w += links[i].calcAverageVal(obs);
    }

    return w / 9;
  }
};

template <class T>
static std::vector<double> normalizeVector(const std::vector<T> vals, const bool log_likelihood = true)
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
                                            const std::vector<double> normalized_weights,
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

static SpiderParticle jitterParticle(const SpiderParticle& particle, const float jitter_pix, const float jitter_angle)
{
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> dpix{0, jitter_pix};
  std::normal_distribution<float> dangle{0, jitter_angle};

  std::vector<float> new_joints;
  for (auto& j : particle.joints)
  {
    new_joints.push_back(j + dangle(gen));
  }

  std::normal_distribution<float> dx{particle.x, jitter_pix};
  std::normal_distribution<float> dy{particle.y, jitter_pix};
  SpiderParticle new_particle(particle.x + dpix(gen), particle.y + dpix(gen), new_joints);

  return new_particle;
}

static std::vector<SpiderParticle> jitterParticles(const std::vector<SpiderParticle>& particles,
                                                   const float jitter_pix, const float jitter_angle)
{
  std::vector<SpiderParticle> new_particles;

  for (auto& p : particles)
  {
    new_particles.push_back(jitterParticle(p, jitter_pix, jitter_angle));
  }

  return new_particles;
}

};  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H
