#ifndef __CUDAFUNCTIONS_HPP
#define __CUDAFUNCTIONS_HPP

#include "utils/common.hpp"

__global__
void rgb2hsvKernel(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height);

__host__
float rgb2hsvCompute(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height);

#endif