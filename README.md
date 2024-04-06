currently, main.cu is a mandelbrot set visualization.

The mandelbrot set works by choosing a color based off how quickly input to the equation

zn+1=z2n+z0

goes to infinity.

I don't totally understand the mathematical underpinnings of the equation, so there is more info on this page: https://math.libretexts.org/Bookshelves/Analysis/Complex_Analysis_-_A_Visual_and_Interactive_Introduction_(Ponce_Campuzano)/05%3A_Chapter_5/5.05%3A_The_Mandelbrot_Set

Technically this was a very interesting project because I got to observe the limitations of floating point representations (floats and doubles) as well as their ability to represent much smaller numbers than equally large integer number structures (e.g. ints and longs) due to their exponent + mantisa structure.

Also, determining which aspect of the program was preventing greater zooms (e.g. available color representation vs floating point precision) was interesting to debug.

And it was my first introduction to SDL/OpenGL/Graphics and CUDA!
