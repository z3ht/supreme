#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>


__device__ Uint32 getColor(int iter) {
    int r = (iter % 256);
    int g = ((iter * 2) % 256);
    int b = ((iter * 3) % 256);
    return (0xFF << 24) | (r << 16) | (g << 8) | b;
}

__global__ void mandelbrotSetKernel(unsigned int* output, long long lowerX, long long lowerY, long long stepX, long long stepY, int maxIter, int width, int height, long long scaleFactor) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);
    if (idx >= width || idy >= height) return;
    int pixelIndex = (idy * width) + idx;

    long long x0 = lowerX + idx * stepX;
    long long y0 = lowerY + idy * stepY;
    long long x = 0;
    long long y = 0;

    int iteration = 0;
    long long inf = 4LL * scaleFactor * scaleFactor;
    while (x*x + y*y <= inf && iteration < maxIter) {
        long long xtemp = x*x/scaleFactor - y*y/scaleFactor + x0;
        y = 2*x*y/scaleFactor + y0;
        x = xtemp;
        iteration++;
    }

    output[pixelIndex] = getColor(iteration);
}


int main() {
    long long scaleFactor = 100000000LL;
    const int width = 2024, height = 1568, maxIter = 400;
    long long lowerX = static_cast<long long>(-2.5) * scaleFactor, lowerY = static_cast<long long>(-1.0) * scaleFactor,
              upperX = static_cast<long long>(1.0) * scaleFactor, upperY = static_cast<long long>(1.0) * scaleFactor;

    long long centerX = 0;
    long long centerY = 0;

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
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            } else if (event.type == SDL_MOUSEWHEEL) {
                // Adjust zoomSpeed as a floating-point since it's a multiplicative factor, not an absolute coordinate
                if (event.wheel.y > 0) { zoomSpeed *= zoomSpeed > 1 ? 0.996 : 0.998; }
                else if (event.wheel.y < 0) { zoomSpeed /= zoomSpeed > 1 ? 0.998 : 0.996; }
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);

                centerX = lowerX + ((static_cast<long long>(mouseX) * (upperX - lowerX)) / width);
                centerY = lowerY + ((static_cast<long long>(mouseY) * (upperY - lowerY)) / height);
            }
        }

        long long rangeX = (upperX - lowerX) * zoomSpeed;
        long long rangeY = (upperY - lowerY) * zoomSpeed;

        lowerX = centerX - rangeX / 2;
        upperX = centerX + rangeX / 2;
        lowerY = centerY - rangeY / 2;
        upperY = centerY + rangeY / 2;

        long long stepX = rangeX / width;
        long long stepY = rangeY / height;

        dim3 blocks(16, 16);
        dim3 grid((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1) / blocks.y);

        double unscaledRange = static_cast<double>(upperX - lowerX) / scaleFactor;
        int dynamicMaxIter = static_cast<int>(maxIter * std::max(0.1, 1 - log(unscaledRange) / log(10000)));

        mandelbrotSetKernel<<<grid, blocks>>>(d_output, lowerX, lowerY, stepX, stepY, dynamicMaxIter, width, height, scaleFactor);
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
    delete[] d_output;

    delete[] pixels;

    return 0;
}
