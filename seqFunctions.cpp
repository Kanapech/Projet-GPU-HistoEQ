#include <iostream>
#include <cstdlib>
#include <cstring>

#include "seqFunctions.hpp"
using namespace std;

void rgb2hsvCPU(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height){
    int size = height*width;

    int *red, *green, *blue;

    red = (int*) malloc(size*sizeof(int));
    green = (int*) malloc(size*sizeof(int));
    blue = (int*) malloc(size*sizeof(int));
    int j;
    for(int i = 0, j = 0; i<size; i++, j+=3){
        red[i] = pixels[j];
        green[i] = pixels[j+1];
        blue[i] = pixels[j+2];

        float r, g, b;
        r = red[i] / 255.;
        g = green[i] / 255.;
        b = blue[i] / 255.;

        float cmax = std::max(std::max(r, g), b);
        float cmin = std::min(std::min(r , g), b);
        float diff = cmax-cmin;

        if(diff == 0)
            htab[i] = 0;
        else if(cmax == r)
            htab[i] = (60 * ((g-b)/ diff) + 360) - ((int) ((60 * ((g-b)/ diff) + 360) / 360)) * 360; // fmod = numer - tquot * denom
        else if(cmax == g)
            htab[i] = (60 * ((b-r)/ diff) + 120) - ((int) ((60 * ((b-r)/ diff) + 120) / 360)) * 360;
        else if(cmax == b)
            htab[i] = (60 * ((r-g)/ diff) + 240) - ((int) ((60 * ((r-g)/ diff) + 240) / 360)) * 360;

        if(cmax == 0)
            stab[i] = 0;
        else
            stab[i] = 1 - (cmin/cmax);

        vtab[i] = cmax;
    }
}

unsigned char* hsv2rgbCPU(float* htab, float* stab, float* vtab, int width, int height){
    int size = height*width;
    unsigned char* pixels;
    pixels = (unsigned char*) malloc(size*3*sizeof(unsigned char));
    cout << sizeof(pixels) << endl;
    float c, x, m;
    float h, s, v;
    float r, g, b;

    int j;
    for(int i = 0, j = 0; i<size; i++, j+=3){
        
        h = htab[i]; s = stab[i]; v = vtab[i];
        
        c = v*s;
        x = c * (1 - abs(h/60.f - (int) ((h/60.f)/2) *2 -1));
        m = v-c;

        if(h >= 0 && h < 60){
            r = c;
            g = x;
            b = 0;
        }   
        else if(h >= 60 && h < 120){
            r = x;
            g = c;
            b = 0;
        }
        else if(h >= 120 && h < 180){
            r = 0;
            g = c;
            b = x;
        }          
        else if(h >= 180 && h < 240){
            r = 0;
            g = x;
            b = c;
        }         
        else if(h >= 240 && h < 300){
            r = x;
            g = 0;
            b = c;
        }
        else if(h >= 300 && h < 360){
            r = c;
            g = 0;
            b = x;
        }

        pixels[j] = (r+m)*255;
        pixels[j+1] = (g+m)*255;
        pixels[j+2] = (b+m)*255;

    }

    return pixels;
}