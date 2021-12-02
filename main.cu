#include <iostream>
#include <cstdlib>
#include <cstring>

#include "image.hpp"
#include "seqFunctions.hpp"
#include "utils/chronoCPU.hpp"

using namespace std;

int main(int argc, char **argv){
    Image img;
    img.load("img/Chateau.png");

    int size = img._height*img._width;
    float htab[size], stab[size], vtab[size];

    //cout << strlen((char*)img._pixels) << endl;
    ChronoCPU chr;
	chr.start();
    rgb2hsvCPU(img._pixels, htab, stab, vtab, img._width, img._height);
    chr.stop();
    cout << "rgb2hsvCPU : " << chr.elapsedTime() << "ms" << endl;

	chr.start();
    unsigned char* pix = hsv2rgbCPU(htab, stab, vtab, img._width, img._height);
    chr.stop();
    cout << "hsv2rgbCPU : " << chr.elapsedTime() << "ms" << endl;

    Image im(img._width, img._height, 3);
    im._pixels = pix;
    im.save("output.png");

	chr.start();
    int* hist = histogramCPU(vtab, size);
    chr.stop();
    cout << "histoCPU : " << chr.elapsedTime() << "ms" << endl;

    return 0;
}