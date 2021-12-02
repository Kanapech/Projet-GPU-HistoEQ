#ifndef __SEQFUNCTIONS_HPP
#define __SEQFUNCTIONS_HPP

#include "image.hpp"

void rgb2hsvCPU(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height);

unsigned char* hsv2rgbCPU(float* htab, float* stab, float* vtab, int width, int height);

int* histogramCPU(float* vtab, int size);

#endif