#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <SDL.h>

#include "gui.h"

#define THREADS 32

// Time step size
#define DT 0.075

// Gravitational constant
#define G 100

// Relevent radii
#define CANNONBALL_RADIUS 5
#define SPACESHIP_RADIUS 10

// Relevant masses
#define CANNONBALL_MASS 10
#define SPACESHIP_MASS 30

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// This struct holds data for a single cannonball
typedef struct cannonball {
  float x_position;
  float y_position;
  float x_velocity;
  float y_velocity;
} cannonball_t;

// This struct holds data for a user's spaceship
typedef struct spaceship {
  int clientID;
  float x_position;
  float y_position;
  float x_velocity;
  float y_velocity;
} spaceship_t;

// This struct holds data for a star
typdef struct star {
  float mass;
  float x_position;
  float y_position;
} star_t;

// This isn't needed because cannonball size isn't random
/*
float drand(float min, float max) {
  return ((float)rand() / RAND_MAX) * (max - min) + min;
}
*/

// Compute the radius of a star based on its mass

__device__ __host__ float star_radius(float mass) {
  return sqrt(mass);
}


__global__ void updateCannonballs(cannonball_t* cannonballs, star_t* stars, int num_cannonballs, int num_stars) {
  int i = (blockIdx.x * THREADS) + threadIdx.x;
  if (i < num_cannonballs) {
    cannonballs[i].x_position += cannonballs[i].x_velocity * DT;
    cannonballs[i].y_position += cannonballs[i].y_velocity * DT;

    // Loop over all other cannonballs to compute forces
    for(int j = 0; j < num_stars ; j++) {
      // Don't compute the force of a star on itself
      // vvv cannonballs don't compute on themselves
      // if(i == j) continue;

      // Compute the distance between the cannonball and each star in each dimension
      float x_diff = cannonballs[i].x_position - stars[j].x_position;
      float y_diff = cannonballs[i].y_position - stars[j].y_position;

      // Compute the magnitude of the distance vector
      float dist = sqrt(x_diff * x_diff + y_diff * y_diff);

      // Normalize the distance vector components
      x_diff /= dist;
      y_diff /= dist;

      // Keep a minimum distance, otherwise we get
      // Is this necessary? Could be used for collisions
      float combined_radius = CANNONBALL_RADIUS + star_radius(stars[j].mass);
      if(dist < combined_radius) {
        dist = combined_radius;
      }

      // Compute the x and y accelerations
      float x_acceleration = -x_diff * G * CANNONBALL_MASS / (dist * dist);
      float y_acceleration = -y_diff * G * CANNONBALL_MASS / (dist * dist);

      // Update the star velocity
      cannonballs[i].x_velocity += x_acceleration * DT;
      cannonballs[i].y_velocity += y_acceleration * DT;

      // Handle edge collisiosn
      if(cannonballs[i].x_position < 0 && cannonballs[i].x_velocity < 0) cannonballs[i].x_velocity *= -0.5;
      if(cannonballs[i].x_position >= SCREEN_WIDTH && cannonballs[i].x_velocity > 0) cannonballs[i].x_velocity *= -0.5;
      if(cannonballs[i].y_position < 0 && cannonballs[i].y_velocity < 0) cannonballs[i].y_velocity *= -0.5;
      if(cannonballs[i].y_position >= SCREEN_HEIGHT && cannonballs[i].y_velocity > 0) cannonballs[i].y_velocity *= -0.5;
    }
  }
}

int main(int argc, char** argv) {
  // Initialize the graphical interface
  gui_init();

  // Run as long as this is true
  bool running = true;

  // Is the mouse currently clicked?
  bool clicked = false;

  // This will hold our array of cpu_stars
  star_t* cpu_stars = NULL;
  star_t* gpu_stars = NULL;
  int num_stars = 0;

  // Start main loop
  while(running) {
    // Check for events
    SDL_Event event;
    while(SDL_PollEvent(&event) == 1) {
      // If the event is a quit event, then leave the loop
      if(event.type == SDL_QUIT) running = false;
    }

    // Get the current mouse state
    int mouse_x, mouse_y;
    uint32_t mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);

    // Is the mouse pressed?
    if(mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      // Is this the beginning of a mouse click?
      if(!clicked) {
        clicked = true;
        cpu_stars = (star_t*)realloc(cpu_stars, (num_stars + 1) * sizeof(star_t));
        cpu_stars[num_stars].x_position = mouse_x;
        cpu_stars[num_stars].y_position = mouse_y;
        cpu_stars[num_stars].x_velocity = 0;
        cpu_stars[num_stars].y_velocity = 0;
        // Generate a random mass skewed toward small sizes
        cpu_stars[num_stars].mass = drand(0, 1) * drand(0, 1) * 50;
        num_stars++;

        // Copy to the GPU
        if (gpu_stars != NULL) {
          cudaFree(gpu_stars);
        }
        gpuErrchk(cudaMalloc(&gpu_stars, sizeof(star_t) * (num_stars + 1)));

	gpuErrchk(cudaMemcpy(gpu_stars, cpu_stars, sizeof(star_t) * (num_stars + 1), cudaMemcpyHostToDevice));

        // Remember to free gpu_stars!
      }
    } else {
      // The mouse click is finished
      clicked = false;
    }

    // Draw stars
    for(int i=0; i<num_stars; i++) {
      color_t color = {255, 255, 255, 255};
      gui_draw_circle(cpu_stars[i].x_position, cpu_stars[i].y_position, star_radius(cpu_stars[i].mass), color);
    }

    int blocks = (num_stars + THREADS - 1) / THREADS;

    updateStars<<<blocks, THREADS>>>(gpu_stars, num_stars);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy udated stars back to CPU
    gpuErrchk(cudaMemcpy(cpu_stars, gpu_stars, sizeof(star_t) * num_stars, cudaMemcpyDeviceToHost));

  }

  // Free the stars array
  free(cpu_stars);
  cudaFree(gpu_stars);

  // Clean up the graphical interface
  gui_shutdown();

  return 0;
}
