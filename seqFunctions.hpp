#ifndef __SEQFUNCTIONS_HPP
#define __SEQFUNCTIONS_HPP

#include "image.hpp"

float rgb2hsvCPU(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height);

float hsv2rgbCPU(float* htab, float* stab, float* vtab, unsigned char* pixels, int width, int height);

float histogramCPU(float* vtab, int* hist, int width, int height);

float repartCPU(int* hist, int* repart);

float equalizationCPU(int* repart, float* vtab, float* eqVtab, int width, int height);

#endif