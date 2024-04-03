#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include <cgbn/cgbn.h>


__device__ Uint32 getColor(int iter) {
    int r = (iter % 256);
    int g = ((iter * 2) % 256);
    int b = ((iter * 3) % 256);
    return (0xFF << 24) | (r << 16) | (g << 8) | b;
}

__global__ void mandelbrotSetKernel(unsigned int* output, double lowerX, double lowerY, double stepX, double stepY, int maxIter, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= width || idy >= height) return;
    int pixelIndex = (idy * width) + idx;

    double x0 = lowerX + idx * stepX;
    double y0 = lowerY + idy * stepY;
    double x = 0.0;
    double y = 0.0;

    int iteration = 0;
    while (x*x + y*y <= (2*2) && iteration < maxIter) {
        double xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iteration++;
    }

    output[pixelIndex] = getColor(iteration);
}


int main() {
    const int width = 2024, height = 1568, maxIter = 400;
    double lowerX = -2.5, lowerY = -1.0, upperX = 1.0, upperY = 1.0;

    double centerX = -1.186592f;
    double centerY = -1.901211e-1f;

    double zoomSpeed = 1; // Initial zoom speed; smaller values zoom in faster.

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Could not initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("Mandelbrot Set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    SDL_Surface* surface = SDL_GetWindowSurface(window);

    bool quit = false;
    SDL_Event event;

    unsigned int* d_output;
    cudaMalloc(&d_output, width * height * sizeof(unsigned int));

    unsigned int* pixels = new unsigned int[width * height];

    while (!quit) {
        float zoomFactor = 1;

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            } else if (event.type == SDL_MOUSEWHEEL) {
                if (event.wheel.y > 0) { zoomSpeed *= zoomSpeed > 1 ? 0.996 : 0.998; }
                else if (event.wheel.y < 0) { zoomSpeed /= zoomSpeed > 1 ? 0.998 : 0.996; }
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);

                centerX = lowerX + (mouseX / (double)width) * (upperX - lowerX);
                centerY = lowerY + (mouseY / (double)height) * (upperY - lowerY);
            }
        }

        double rangeX = (upperX - lowerX) * zoomSpeed;
        double rangeY = (upperY - lowerY) * zoomSpeed;

        lowerX = centerX - rangeX / 2;
        upperX = centerX + rangeX / 2;
        lowerY = centerY - rangeY / 2;
        upperY = centerY + rangeY / 2;

        float stepX = (upperX - lowerX) / width;
        float stepY = (upperY - lowerY) / height;

        dim3 blocks(16, 16);
        dim3 grid((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1) / blocks.y);

        double range = upperX - lowerX;
        int dynamicMaxIter = static_cast<int>(maxIter * std::max(0.1, 1 - log(range) / log(10000)));

        mandelbrotSetKernel<<<grid, blocks>>>(d_output, lowerX, lowerY, stepX, stepY, dynamicMaxIter, width, height);
        cudaMemcpy(pixels, d_output, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        SDL_FillRect(surface, NULL, SDL_MapRGB(surface->format, 0, 0, 0));

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < width * height; ++i) {
            ((Uint32*)surface->pixels)[i] = pixels[i];
        }

        SDL_UpdateWindowSurface(window);
    }

    SDL_Quit();

    cudaFree(d_output);
    delete d_output;

    delete pixels;

    return 0;
}
