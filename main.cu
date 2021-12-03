#include <iostream>
#include <cstdlib>
#include <cstring>

#include "image.hpp"
#include "seqFunctions.hpp"
#include "cudaFunctions.hpp"

using namespace std;

int main(int argc, char **argv){
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
    im.save("output.png");

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
    cout << "rgb2hsvCPU : " << timeGPU << "ms" << endl;



    return 0;
}