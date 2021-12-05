#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "seqFunctions.hpp"
#include "utils/chronoCPU.hpp"

using namespace std;

float rgb2hsvCPU(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height){
    ChronoCPU chr;
	chr.start();
    int size = height*width;

    int j;
    for(int i = 0, j = 0; i<size; i++, j+=3){

        float r, g, b;
        r = pixels[j] / 255.;
        g = pixels[j+1] / 255.;
        b = pixels[j+2] / 255.;

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

        /* plus long
        if(cmax == 0)
            stab[i] = 0;
        else
            stab[i] = 1 - (cmin/cmax);
        */

        stab[i] = cmax == 0 ? 0 : 1 - (cmin/cmax);

        vtab[i] = cmax;

    }
    chr.stop();
    return chr.elapsedTime();
}

float hsv2rgbCPU(float* htab, float* stab, float* vtab, unsigned char* pixels, int width, int height){

    ChronoCPU chr;
	chr.start();
    int size = height*width;

    float c, x, m;
    float h, s, v;
    float r, g, b;

    int j;
    for(int i = 0, j = 0; i<size; i++, j+=3){
        
        h = htab[i]; s = stab[i]; v = vtab[i];
        
        c = v*s;
        x = c * (1 - fabs(h/60.f - (int) ((h/60.f)/2) *2 -1));
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
    chr.stop();

    return chr.elapsedTime();
}

float histogramCPU(float* vtab, int* hist, int width, int height){
    ChronoCPU chr;
	chr.start();
    int size = width*height;

    for(int i=0; i<size ; i++)
        hist[(int)(vtab[i] * 100)] ++;

    chr.stop();
    
    return chr.elapsedTime();
}

float repartCPU(int* hist, int* repart){
    ChronoCPU chr;
	chr.start();

    int sum = 0;
    for (int i=0; i<256; i++){
        sum += hist[i]; 
        repart[i] = sum;
    }  

    chr.stop();
    return chr.elapsedTime();
}

float equalizationCPU(int* repart, float* vtab, float* eqVtab, int width, int height){
    ChronoCPU chr;
	chr.start();
    int size = width*height;
    for(int i=0; i<size; i++){
        eqVtab[i] = (float)(255.f/(256*size))*repart[(int)(vtab[i] * 100)];
        //cout << eqVtab[i] << endl;
    }
        

    chr.stop();
    return chr.elapsedTime();
}