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
    file_path_("/home/jana/code/bp-sandbox/media/obs.pbm")
  {
    loadImage(file_path_);

    checked_grid_ = new bool[width * height];
    markUnchecked();
  }

  ~Observation()
  {
    delete checked_grid_;
    delete data_;
  }

  size_t width, height;

  /**
   * Get the pixel value.
   * @param  i                 The column index.
   * @param  j                 The row index.
   * @param  allow_outofbounds If true, out of bounds cells return the lowest value.
   * @param  double_count      If false, cells which have already been accessed return the lowest value.
   * @return                   The pixel value (depends on the flags).
   */
  float getPixel(const int i, const int j,
                 const bool allow_outofbounds = true,
                 const bool double_count = false) const
  {
    // col, row
    if (allow_outofbounds)
    {
      // Returns a 1 if out of bounds.
      if (i < 0 || i >= width || j < 0 || j >= height)
      {
        return 1;
      }
    }

    // Returns a 1 if this was already checked and we don't want to double count cells.
    if (checked_grid_[j * width + i] && !double_count)
    {
      return 1;
    }

    // Mark this one as checked.
    checked_grid_[j * width + i] = true;

    return data_[j * width + i];
  }

  /**
   * Reset the checked grid. Note that this is technically const because it only
   * modifies the checked_grid_ array, which is mutable. This is so that we can
   * decide when to reset the checked grid whenever we change particles.
   */
  void markUnchecked() const
  {
    std::fill(checked_grid_, checked_grid_ + width * height, false);
  }

private:

  float* data_;
  mutable bool* checked_grid_;
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
};

}  // namespace BPSandbox

#endif  // BP_SANDBOX_INFERENCE_COMMON_OBSERVATION_H
