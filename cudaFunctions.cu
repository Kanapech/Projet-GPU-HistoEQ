#include "cudaFunctions.hpp"
#include "utils/chronoGPU.hpp"

__global__
void rgb2hsvKernel(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height){
    int x, y, offset, offsetHSV;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

	offset = (y*width+x)*3;
    offsetHSV = (y*width+x);

    if(x < width && y < height){

        float r, g, b;
        r = pixels[offset] / 255.;
        g = pixels[offset+1] / 255.;
        b = pixels[offset+2] / 255.;

        float cmax = max(max(r, g), b);
        float cmin = min(min(r , g), b);
        float diff = cmax-cmin;

        if(diff == 0)
            htab[offsetHSV] = 0;
        else if(cmax == r)
            htab[offsetHSV] = (60 * ((g-b)/ diff) + 360) - ((int) ((60 * ((g-b)/ diff) + 360) / 360)) * 360; // fmod = numer - tquot * denom
        else if(cmax == g)
            htab[offsetHSV] = (60 * ((b-r)/ diff) + 120) - ((int) ((60 * ((b-r)/ diff) + 120) / 360)) * 360;
        else if(cmax == b)
            htab[offsetHSV] = (60 * ((r-g)/ diff) + 240) - ((int) ((60 * ((r-g)/ diff) + 240) / 360)) * 360;

        if(cmax == 0)
            stab[offsetHSV] = 0;
        else
            stab[offsetHSV] = 1 - (cmin/cmax);

        vtab[offsetHSV] = cmax;

    }
}

__host__
float rgb2hsvCompute(unsigned char* pixels, float* htab, float* stab, float* vtab, int width, int height){
    unsigned char* dev_pixels;
    float *dev_htab, *dev_stab, *dev_vtab;

    int size = width*height;
    // Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&dev_pixels, size*3*sizeof(unsigned char)));
		HANDLE_ERROR(cudaMalloc(&dev_htab, size*sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&dev_stab, size*sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&dev_vtab, size*sizeof(float)));
    
    // Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_pixels, pixels, size*3*sizeof(unsigned char), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_htab, htab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_stab, stab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_vtab, vtab, size*sizeof(float), cudaMemcpyHostToDevice));

    //Kernel settings
        dim3 blockDim(32, 32);
        dim3 gridDim( width / blockDim.x, height / blockDim.y); 
        
    ChronoGPU chr;
	chr.start();

	// Launch kernel
		rgb2hsvKernel<<<gridDim, blockDim>>>(dev_pixels, dev_htab, dev_stab, dev_vtab, width, height);
	
	chr.stop();

	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(pixels, dev_pixels, size*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(htab, dev_htab, size*sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(stab, dev_stab, size*sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(vtab, dev_vtab, size*sizeof(float), cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_pixels));
		HANDLE_ERROR(cudaFree(dev_htab));
        HANDLE_ERROR(cudaFree(dev_stab));
        HANDLE_ERROR(cudaFree(dev_vtab));

	return chr.elapsedTime();
}

__global__
void hsv2rgbKernel(float* htab, float* stab, float* vtab, unsigned char* pixels, int width, int height){
    int xIndex, yIndex, offset, offsetHSV;
    xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	offset = (yIndex*width+xIndex)*3;
    offsetHSV = (yIndex*width+xIndex);

    float c, x, m;
    float h, s, v;
    float r, g, b;

    if(xIndex < width && yIndex < height){
        h = htab[offsetHSV]; s = stab[offsetHSV]; v = vtab[offsetHSV];
        
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

        pixels[offset] = (r+m)*255;
        pixels[offset+1] = (g+m)*255;
        pixels[offset+2] = (b+m)*255;
    }


}

__host__
float hsv2rgbCompute(float* htab, float* stab, float* vtab, unsigned char* pixels, int width, int height){
    unsigned char* dev_pixels;
    float *dev_htab, *dev_stab, *dev_vtab;

    int size = width*height;
    // Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&dev_pixels, size*3*sizeof(unsigned char)));
		HANDLE_ERROR(cudaMalloc(&dev_htab, size*sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&dev_stab, size*sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&dev_vtab, size*sizeof(float)));
    
    // Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_pixels, pixels, size*3*sizeof(unsigned char), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_htab, htab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_stab, stab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_vtab, vtab, size*sizeof(float), cudaMemcpyHostToDevice));

    //Kernel settings
        dim3 blockDim(32, 32);
        dim3 gridDim( width / blockDim.x, height / blockDim.y); 
        
    ChronoGPU chr;
	chr.start();

	// Launch kernel
		hsv2rgbKernel<<<gridDim, blockDim>>>(dev_htab, dev_stab, dev_vtab, dev_pixels, width, height);
	
	chr.stop();

	// Copy from Device to Host
        HANDLE_ERROR(cudaMemcpy(pixels, dev_pixels, size*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(htab, dev_htab, size*sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(stab, dev_stab, size*sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(vtab, dev_vtab, size*sizeof(float), cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_pixels));
		HANDLE_ERROR(cudaFree(dev_htab));
        HANDLE_ERROR(cudaFree(dev_stab));
        HANDLE_ERROR(cudaFree(dev_vtab));

	return chr.elapsedTime();
}