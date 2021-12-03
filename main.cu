#include <iostream>
#include <cstdlib>
#include <cstring>

#include "image.hpp"
#include "seqFunctions.hpp"
#include "cudaFunctions.hpp"

using namespace std;

bool compare_float(float x, float y, float epsilon = 0.01f){
   if(fabs(x - y) < epsilon)
      return true; //they are same
      return false; //they are not same
}

int main(int argc, char **argv){

    cout << "======= Sequential on CPU ==========" << endl;
    float timeCPU;
    Image img;
    img.load("img/Chateau.png");

    int size = img._height*img._width;
    float htab[size], stab[size], vtab[size];
        timeCPU = rgb2hsvCPU(img._pixels, htab, stab, vtab, img._width, img._height);
    cout << "rgb2hsvCPU : " << timeCPU << "ms" << endl;


    unsigned char* pixels = (unsigned char*) malloc(size*3*sizeof(unsigned char));
        timeCPU =hsv2rgbCPU(htab, stab, vtab, pixels, img._width, img._height);
    cout << "hsv2rgbCPU : " << timeCPU << "ms" << endl;

    Image im(img._width, img._height, 3);
    im._pixels = pixels;
    im.save("outputCPU.png");

    int* hist = (int*) calloc(256, sizeof(int));
        timeCPU = histogramCPU(vtab, hist, size);
    cout << "histoCPU : " << timeCPU << "ms" << endl;




    cout << "======= Parallel on GPU ==========" << endl;
    float timeGPU;
    size = img._height*img._width;
    float *htabGPU = new float[size]; 
    float *stabGPU = new float[size];
    float *vtabGPU = new float[size];
        timeGPU = rgb2hsvCompute(pixels, htabGPU, stabGPU, vtabGPU, img._width, img._height);
    cout << "rgb2hsvGPU : " << timeGPU << "ms" << endl;

    int i=0;
    while(compare_float(htab[i], htabGPU[i]) && i < size)
        i++;
    cout << i << endl;
    cout << htab[i] << " " << htabGPU[i] << endl;

    unsigned char* pixelsGPU = (unsigned char*) malloc(size*3*sizeof(unsigned char));
    timeGPU = hsv2rgbCompute(htabGPU, stabGPU, vtabGPU, pixelsGPU, img._width, img._height);
    cout << "hsv2rgbGPU : " << timeGPU << "ms" << endl;
    im._pixels = pixelsGPU;
    im.save("outputGPU.png");

    i=0;
    while(pixels[i] == pixelsGPU[i] && i < size*3)
        i++;
    cout << i << endl;
    cout << pixels[i] << " " << pixelsGPU[i] << endl;

    return 0;
}