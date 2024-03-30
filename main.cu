#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>

__global__ void mandelbrotSetKernel(unsigned int* output, float lowerX, float lowerY, float stepX, float stepY, int maxIter, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= width || idy >= height) return;
    int pixelIndex = (idy * width) + idx;

    float x0 = lowerX + idx * stepX;
    float y0 = lowerY + idy * stepY;
    float x = 0.0;
    float y = 0.0;

    int iteration = 0;
    while (x*x + y*y <= (2*2) && iteration < maxIter) {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iteration++;
    }

    output[pixelIndex] = iteration;
}

Uint32 getColor(int iter) {
    int r = (iter % 256);
    int g = ((iter * 2) % 256);
    int b = ((iter * 3) % 256);
    return (0xFF << 24) | (r << 16) | (g << 8) | b;
}


int main() {
    const int width = 1024, height = 768, maxIter = 1000000;
    float lowerX = -2.5, lowerY = -1.0, upperX = 1.0, upperY = 1.0;

    float centerX = -1.186592f;
    float centerY = -1.901211e-1f;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Could not initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("Mandelbrot Set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    SDL_Surface* surface = SDL_GetWindowSurface(window);

    bool quit = false;
    SDL_Event event;

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            } else if (event.type == SDL_MOUSEWHEEL) {
                float zoomFactor = (event.wheel.y > 0) ? 0.9f : 1.1f;
                float rangeX = (upperX - lowerX) * zoomFactor;
                float rangeY = (upperY - lowerY) * zoomFactor;
                // float centerX = (upperX + lowerX) / 2;
                // float centerY = (upperY + lowerY) / 2;

                lowerX = centerX - rangeX / 2;
                upperX = centerX + rangeX / 2;
                lowerY = centerY - rangeY / 2;
                upperY = centerY + rangeY / 2;
            }

        }

        float stepX = (upperX - lowerX) / width;
        float stepY = (upperY - lowerY) / height;

        unsigned int* d_output;
        cudaMalloc(&d_output, width * height * sizeof(unsigned int));
        dim3 blocks(32, 32); // Corrected block dimensions
        dim3 grid((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1) / blocks.y); // Adjusted for corrected blocks
        mandelbrotSetKernel<<<grid, blocks>>>(d_output, lowerX, lowerY, stepX, stepY, maxIter, width, height);

        unsigned int* pixels = new unsigned int[width * height];
        cudaMemcpy(pixels, d_output, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        SDL_FillRect(surface, NULL, SDL_MapRGB(surface->format, 0, 0, 0));

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < width * height; ++i) {
            unsigned int iter = pixels[i];
            Uint32 pixelColor = getColor(iter);
            ((Uint32*)surface->pixels)[(i / width) * surface->w + (i % width)] = pixelColor;
        }

        SDL_UpdateWindowSurface(window);

        cudaFree(d_output);
        delete[] pixels;
    }

    SDL_Quit();

    return 0;
}
