#ifndef BP_SANDBOX_INFERENCE_COMMON_OBSERVATION_H
#define BP_SANDBOX_INFERENCE_COMMON_OBSERVATION_H

#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace BPSandbox
{

class Observation
{
public:
  Observation() :
    file_path_("/home/jana/code/bp-sandbox/data/obs.pbm"),
    num_occupied(0)
  {
    loadImage(file_path_);
  }

  ~Observation()
  {
    delete data_;
  }

  size_t width, height;
  int num_occupied;

  /**
   * Get the pixel value.
   * @param  i                 The column index.
   * @param  j                 The row index.
   * @param  allow_outofbounds If true, out of bounds cells return the lowest value.
   * @return                   The pixel value (depends on the flags).
   */
  float getPixel(const int i, const int j,
                 const bool allow_outofbounds = true) const
  {
    // col, row
    if (allow_outofbounds)
    {
      // Returns a 0 if out of bounds.
      if (i < 0 || i >= width || j < 0 || j >= height)
      {
        return 0;
      }
    }

    return data_[j * width + i];
  }

  bool isOccupied(const int i, const int j) const
  {
    return data_[j * width + i] == 1.0;
  }

  void setPixel(const int i, const int j, const float val)
  {
    data_[j * width + i] = val;
  }

private:

  float* data_;
  std::string file_path_;

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
    // The second line has width and height.
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
        if (isOccupied(col, row)) num_occupied++;
      }
    }
  }
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_OBSERVATION_H
