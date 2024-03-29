#include "cudaFunctions.hpp"
#include "utils/chronoGPU.hpp"

//----------------------- rgb2hsv ----------------------------//

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

        stab[offsetHSV] = cmax == 0 ? 0 : 1 - (cmin/cmax);
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

    //Kernel settings
        dim3 blockDim(32, 32);
        dim3 gridDim( (width-1)/blockDim.x+1, (height-1)/blockDim.y+1); 
        
    ChronoGPU chr;
	chr.start();

	// Launch kernel
		rgb2hsvKernel<<<gridDim, blockDim>>>(dev_pixels, dev_htab, dev_stab, dev_vtab, width, height);
	
	chr.stop();

	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(htab, dev_htab, size*sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(stab, dev_stab, size*sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(vtab, dev_vtab, size*sizeof(float), cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_htab));
        HANDLE_ERROR(cudaFree(dev_stab));
        HANDLE_ERROR(cudaFree(dev_vtab));

	return chr.elapsedTime();
}

//----------------------- hsv2rgb ----------------------------//

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
		HANDLE_ERROR(cudaMemcpy(dev_htab, htab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_stab, stab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_vtab, vtab, size*sizeof(float), cudaMemcpyHostToDevice));

    //Kernel settings
        dim3 blockDim(32, 32);
        dim3 gridDim((width-1)/blockDim.x+1, (height-1)/blockDim.y+1); 
        
    ChronoGPU chr;
	chr.start();

	// Launch kernel
		hsv2rgbKernel<<<gridDim, blockDim>>>(dev_htab, dev_stab, dev_vtab, dev_pixels, width, height);
	
	chr.stop();

	// Copy from Device to Host
        HANDLE_ERROR(cudaMemcpy(pixels, dev_pixels, size*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_pixels));
		HANDLE_ERROR(cudaFree(dev_htab));
        HANDLE_ERROR(cudaFree(dev_stab));
        HANDLE_ERROR(cudaFree(dev_vtab));

	return chr.elapsedTime();
}

//----------------------- histo ----------------------------//

__global__
void histoKernel(float* vtab, int* hist, int width, int height){
    __shared__ int histo[256];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int rang = threadIdx.x;

    if(rang < 256){
        histo[rang] = 0;
    }
    __syncthreads();

    if(pos<width*height){
        atomicAdd(histo+(int)(vtab[pos] * 100), 1);
    }
    __syncthreads();

    if(rang < 256){
        atomicAdd(hist+rang, histo[rang]);
    }

    /* version plus lente (plus de 5x) 
    int tid, size;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    size = width*height;

    if(tid < size)
        atomicAdd(hist+(int)(vtab[tid] * 100), 1);
    */
}

__host__
float histoCompute(float* vtab, int* hist, int width, int height){
    int* dev_hist;
    float *dev_vtab;
    int size = width*height;

    // Allocate memory on Device
        HANDLE_ERROR(cudaMalloc(&dev_hist, 256*sizeof(int)));
        HANDLE_ERROR(cudaMalloc(&dev_vtab, size*sizeof(float)));
    
    // Copy from Host to Device
        HANDLE_ERROR(cudaMemcpy(dev_vtab, vtab, size*sizeof(float), cudaMemcpyHostToDevice));

    //Kernel settings
        int threads = 1024;
        int blocks = (size + threads-1) / threads;

    ChronoGPU chr;
	chr.start();

	// Launch kernel
		histoKernel<<<blocks, threads>>>(dev_vtab, dev_hist, width, height);
	
	chr.stop();

	// Copy from Device to Host
        HANDLE_ERROR(cudaMemcpy(hist, dev_hist, 256*sizeof(int), cudaMemcpyDeviceToHost));

	// Free memory on Device
        HANDLE_ERROR(cudaFree(dev_hist));
        HANDLE_ERROR(cudaFree(dev_vtab));

	return chr.elapsedTime();
}

//----------------------- repart ----------------------------//

__global__
void repartKernel(int* hist, int* repart){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sharedRepart[256];

    //Remplissage du tableau partagé avec les valeurs de l'histogramme
    //if(tid < 256) Pas besoin car 256 threads
    sharedRepart[threadIdx.x] = hist[threadIdx.x];
    __syncthreads();

    //Calcul de la fonction de repartition
    for (int offset = 1; offset < 256; offset *= 2){ 
        if (threadIdx.x >= offset) {
            sharedRepart[threadIdx.x] += sharedRepart[threadIdx.x - offset]; 	
        }
        __syncthreads(); 
    }

    //Remplissage du tableau retourné
    //if(tid < 256) Pas besoin car 256 threads
    repart[tid] = sharedRepart[threadIdx.x];
}

__host__
float repartCompute(int* hist, int* repart){
    int *dev_hist, *dev_repart;;

    // Allocate memory on Device
        HANDLE_ERROR(cudaMalloc(&dev_hist, 256*sizeof(int)));
        HANDLE_ERROR(cudaMalloc(&dev_repart, 256*sizeof(int)));
    
    // Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_hist, hist, 256*sizeof(int), cudaMemcpyHostToDevice));

    //Kernel settings
        int nbThreads = 256;
        int nbBlocks = 1;
        
    ChronoGPU chr;
	chr.start();

	// Launch kernel
		repartKernel<<<nbBlocks, nbThreads>>>(dev_hist, dev_repart);
	
	chr.stop();

	// Copy from Device to Host
        HANDLE_ERROR(cudaMemcpy(repart, dev_repart, 256*sizeof(int), cudaMemcpyDeviceToHost));

	// Free memory on Device
        HANDLE_ERROR(cudaFree(dev_hist));
        HANDLE_ERROR(cudaFree(dev_repart));

	return chr.elapsedTime();
}

__global__
void equalizationKernel(int* repart, float* vtab, float* eqVtab, int width, int height){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width*height;

    if(tid < size)
        eqVtab[tid] = (float)(255.f/(256*size))*repart[(int)(vtab[tid] * 100)];
}

__host__
float equalizationCompute(int* repart, float* vtab, float* eqVtab, int width, int height){
    int* dev_repart;
    float *dev_vtab, *dev_eqVtab;
    int size = width*height;

    // Allocate memory on Device
        HANDLE_ERROR(cudaMalloc(&dev_repart, 256*sizeof(int)));
        HANDLE_ERROR(cudaMalloc(&dev_vtab, size*sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&dev_eqVtab, size*sizeof(float)));
    
    // Copy from Host to Device
        HANDLE_ERROR(cudaMemcpy(dev_repart, repart, 256*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_vtab, vtab, size*sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_eqVtab, eqVtab, size*sizeof(float), cudaMemcpyHostToDevice));

    //Kernel settings
        int threads = 1024;
        int blocks = (size + threads-1) / threads;
        
    ChronoGPU chr;
	chr.start();

	// Launch kernel
		equalizationKernel<<<blocks, threads>>>(dev_repart, dev_vtab, dev_eqVtab, width, height);
	
	chr.stop();

	// Copy from Device to Host
        HANDLE_ERROR(cudaMemcpy(eqVtab, dev_eqVtab, size*sizeof(float), cudaMemcpyDeviceToHost));

	// Free memory on Device
        HANDLE_ERROR(cudaFree(dev_repart));
        HANDLE_ERROR(cudaFree(dev_vtab));
        HANDLE_ERROR(cudaFree(dev_eqVtab));

	return chr.elapsedTime();
}