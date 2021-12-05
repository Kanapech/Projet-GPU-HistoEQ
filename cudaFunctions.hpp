#ifndef __CUDAFUNCTIONS_HPP
#define __CUDAFUNCTIONS_HPP

#include "utils/common.hpp"

// ---------------- Kernels ------------------------------------//

__global__
void rgb2hsvKernel(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height);

__global__
void hsv2rgbKernel(float* htab, float* stab, float* vtab, unsigned char* pixels, int width, int height);

__global__
void histoKernel(float* vtab, int* hist, int width, int height);

__global__
void repartKernel(int* hist, int* repart);

__global__
void equalizationKernel(int* repart, float* vtab, float* eqVtab, int width, int height);


// ---------------- Call functions ------------------------------------//

__host__
float rgb2hsvCompute(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height);

__host__
float hsv2rgbCompute(float* htab, float* stab, float* vtab, unsigned char* pixels, int width, int height);

__host__
float histoCompute(float* vtab, int* hist, int width, int height);

__host__
float repartCompute(int* hist, int* repart);

__host__
float equalizationCompute(int* repart, float* vtab, float* eqVtab, int width, int height);

#endif