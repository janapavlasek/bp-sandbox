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

typedef std::vector<std::vector<float> > ParticleList;

class Observation
{
public:
  Observation() :
    file_path_("/home/jana/code/bp-sandbox/media/obs.pbm")
  {
    loadImage(file_path_);
  }

  float getPixel(const int i, const int j)
  {
    // col, row
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

static inline float normalize_angle(float angle)
{
  const double result = fmod(angle, 2.0*M_PI);
  if(result < 0) return result + 2.0*M_PI;
  return result;
}

};  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_INFERENCE_UTILS_H
