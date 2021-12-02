#include <iostream>
#include <cstdlib>
#include <cstring>

#include "image.hpp"
#include "seqFunctions.hpp"

using namespace std;

int main(int argc, char **argv){
    Image img;
    img.load("img/Chateau.png");

    int size = img._height*img._width;
    float htab[size], stab[size], vtab[size];

    //cout << strlen((char*)img._pixels) << endl;
    rgb2hsvCPU(img._pixels, htab, stab, vtab, img._width, img._height);
    unsigned char* pix = hsv2rgbCPU(htab, stab, vtab, img._width, img._height);

    Image im(img._width, img._height, 3);
    im._pixels = pix;
    im.save("output.png");

    int* hist = histogramCPU(vtab, size);

    return 0;
}