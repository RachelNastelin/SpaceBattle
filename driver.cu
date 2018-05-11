#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <SDL.h>
#include <cuda.h>

#include "board.h"
#include "driver.h"

/***************************************MACRO DEFINITIONS*********************************************/


#define THREADS 32

// Time step size
#define DT 0.075

// Gravitational constant
#define G 100

// Relevent radii
#define CANNONBALL_RADIUS 2
#define SPACESHIP_RADIUS 4

// Relevant masses
#define CANNONBALL_MASS 4
#define SPACESHIP_MASS 16

#define CANNONBALL_EXIT_POS 10
#define CANNONBALL_EXIT_VEL 10

// Directions
#define UP 1
#define DOWN 2
#define RIGHT 3
#define LEFT 4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// This isn't needed because cannonball size isn't random
/*
float drand(float min, float max) {
  return ((float)rand() / RAND_MAX) * (max - min) + min;
}
*/


/***************************************GLOBAL VARIABLES*********************************************/
// These variables should never be modified beyond their initialized values.
star_t* stars;
int num_stars;


/***************************************FUNCTION IMPLEMENTATIONS*********************************************/

// Initialize an array of star_t-s representing the stars on the playing field
__host__ star_t* init_stars() {
  stars = (star_t*) malloc(sizeof(star_t) * 2); // Initializing the global array "stars"

  // First star
  stars[0].mass = 400;
  stars[0].radius = 20;
  stars[0].x_position = SCREEN_WIDTH/3;
  stars[0].y_position = SCREEN_WIDTH/2;
  // Second star
  stars[1].mass = 400;
  stars[1].radius = 20;
  stars[1].x_position = 2*(SCREEN_WIDTH/3);
  stars[1].y_position = SCREEN_WIDTH/2;

  num_stars = 2; // Inititializing the global int "num_stars"

  return stars;
}

// Free the array of stars
__host__ void free_stars() {
  free(stars);
}

// Places the user's spaceship on oneside of the field, depending on whether the user is the first
// or second client to connect to the server
__host__ spaceship_t * init_spaceship(int clientID) {
  spaceship_t* spaceship = (spaceship_t*) malloc(sizeof(spaceship_t));

  spaceship->clientID = clientID;

  switch(clientID) {
    case 0 :
      spaceship->x_position = SCREEN_WIDTH/5;
      spaceship->y_position = SCREEN_HEIGHT/5;
      break;
    case 1 :
      spaceship->x_position = 4*(SCREEN_WIDTH/5);
      spaceship->y_position = 4*(SCREEN_HEIGHT/5);
      break;
  }
  return spaceship;
}

__host__ void free_spaceship(spaceship_t* spaceship) {
  free(spaceship);
}

__host__ cannonball_t* init_cannonballs() {
  cannonball_t* cannonballs = (cannonball_t*) malloc(sizeof(cannonball_t));

  return cannonballs;
}

__host__ void free_cannonballs(cannonball_t* cannonballs) {
  free(cannonballs);
}

__host__ bool is_cannonball_in_bounds(spaceship_t* spaceship, int direction_shot) {
  bool result;
  
  float cannonball_x_pos;
  float cannonball_y_pos;
  float cannonball_x_vel;
  float cannonball_y_vel;

  switch(direction_shot) {
    case UP :
      cannonball_x_pos = spaceship->x_position;
      cannonball_y_pos = spaceship->y_position - CANNONBALL_EXIT_POS;
      cannonball_x_vel = spaceship->x_velocity;
      cannonball_y_vel = spaceship->y_velocity - CANNONBALL_EXIT_VEL;
      break;
    case DOWN :
      cannonball_x_pos = spaceship->x_position;
      cannonball_y_pos = spaceship->y_position + CANNONBALL_EXIT_POS;
      cannonball_x_vel = spaceship->x_velocity;
      cannonball_y_vel = spaceship->y_velocity + CANNONBALL_EXIT_VEL;
      break;
    case RIGHT :
      cannonball_x_pos = spaceship->x_position + CANNONBALL_EXIT_POS;
      cannonball_y_pos = spaceship->y_position;
      cannonball_x_vel = spaceship->x_velocity + CANNONBALL_EXIT_VEL;
      cannonball_y_vel = spaceship->y_velocity;
      break;
    case LEFT :
      cannonball_x_pos = spaceship->x_position - CANNONBALL_EXIT_POS;
      cannonball_y_pos = spaceship->y_position;
      cannonball_x_vel = spaceship->x_velocity - CANNONBALL_EXIT_VEL;
      cannonball_y_vel = spaceship->y_velocity;
      break;
  }

  // Is this cannonball within the bounds of the screen?
  if (cannonball_x_pos > 0 &&
      cannonball_x_pos <= SCREEN_WIDTH &&
      cannonball_y_pos > 0 &&
      cannonball_y_pos <= SCREEN_HEIGHT) {
    result = true;
  }
  else {
    result = false;
  }

  return result;
}

// Add a cannonball to the field (Note: the caller must update the number of cannonballs!)
__host__ cannonball_t* add_cannonball(spaceship_t* spaceship, cannonball_t* cannonballs, int num_cannonballs) {
  float cannonball_x_pos;
  float cannonball_y_pos;
  float cannonball_x_vel;
  float cannonball_y_vel;

  switch(direction_shot) {
    case UP :
      cannonball_x_pos = spaceship->x_position;
      cannonball_y_pos = spaceship->y_position - CANNONBALL_EXIT_POS;
      cannonball_x_vel = spaceship->x_velocity;
      cannonball_y_vel = spaceship->y_velocity - CANNONBALL_EXIT_VEL;
      break;
    case DOWN :
      cannonball_x_pos = spaceship->x_position;
      cannonball_y_pos = spaceship->y_position + CANNONBALL_EXIT_POS;
      cannonball_x_vel = spaceship->x_velocity;
      cannonball_y_vel = spaceship->y_velocity + CANNONBALL_EXIT_VEL;
      break;
    case RIGHT :
      cannonball_x_pos = spaceship->x_position + CANNONBALL_EXIT_POS;
      cannonball_y_pos = spaceship->y_position;
      cannonball_x_vel = spaceship->x_velocity + CANNONBALL_EXIT_VEL;
      cannonball_y_vel = spaceship->y_velocity;
      break;
    case LEFT :
      cannonball_x_pos = spaceship->x_position - CANNONBALL_EXIT_POS;
      cannonball_y_pos = spaceship->y_position;
      cannonball_x_vel = spaceship->x_velocity - CANNONBALL_EXIT_VEL;
      cannonball_y_vel = spaceship->y_velocity;
      break;
  }

  // Only add the cannonball if it will be within the bounds of the screen
  /* This has been moved to func 'is_cannonball_in_bounds'
  if (cannonball_x_pos > 0 &&
      cannonball_x_pos <= SCREEN_WIDTH &&
      cannonball_y_pos > 0 &&
      cannonball_y_pos <= SCREEN_HEIGHT) {
  */

    // Reallocate memory to make space for the new cannonball
  cannonballs = (cannonball_t*)realloc(cannonball, num_cannonballs * sizeof(cannonball_t));
  
    cannonballs[num_cannonballs].x_position = cannonball_x_pos;
    cannonballs[num_cannonballs].y_position = cannonball_y_pos;
    cannonballs[num_cannonballs].x_velocity = cannonball_x_vel;
    cannonballs[num_cannonballs].y_velocity = cannonball_y_vel;
  }

  return cannonballs;
}


// Update position and velocity of a spaceship
__host__ void update_spaceship(spaceship_t* spaceship, int direction_boost) {
  spaceship->x_position += spaceship->x_velocity * DT;
  spaceship->y_position += spaceship->y_velocity * DT;

  // Loop over all stars to compute forces
  for(int j = 0; j < num_stars ; j++) {

    // Compute the distance between the cannonball and each star in each dimension
    float x_diff = spaceship->x_position - stars[j].x_position;
    float y_diff = spaceship->y_position - stars[j].y_position;

    // Compute the magnitude of the distance vector
    float dist = sqrt(x_diff * x_diff + y_diff * y_diff);

    // Normalize the distance vector components
    x_diff /= dist;
    y_diff /= dist;

    // Keep a minimum distance, otherwise we get
    // Is this necessary? Could be used for collisions
    float combined_radius = SPACESHIP_RADIUS + stars[j].radius;
    if(dist < combined_radius) {
      dist = combined_radius;
    }

    // Compute the x and y accelerations
    float x_boost;
    float y_boost;
    switch(direction_boost) {
      case UP :
        x_boost = 0;
        y_boost = -10;
        break;
      case DOWN :
        x_boost = 0;
        y_boost = 10;
        break;
      case RIGHT :
        x_boost = 10;
        y_boost = 0;
        break;
      case LEFT :
        x_boost = -10;
        y_boost = 0;
        break;
    }
    
    float x_acceleration = -x_diff * G * CANNONBALL_MASS / (dist * dist) + x_boost;
    float y_acceleration = -y_diff * G * CANNONBALL_MASS / (dist * dist) + y_boost;

    // Update the star velocity
    spaceship->x_velocity += x_acceleration * DT;
    spaceship->y_velocity += y_acceleration * DT;

    // Handle edge collisiosn
    if(spaceship->x_position < 0 && spaceship->x_velocity < 0) spaceship->x_velocity *= -0.5;
    if(spaceship->x_position >= SCREEN_WIDTH && spaceship->x_velocity > 0) spaceship->x_velocity *= -0.5;
    if(spaceship->y_position < 0 && spaceship->y_velocity < 0) spaceship->y_velocity *= -0.5;
    if(spaceship->y_position >= SCREEN_HEIGHT && spaceship->y_velocity > 0) spaceship->y_velocity *= -0.5;
  }
}

// Has the GPU update cannonballs and transfers them to the CPU.
__host__ void  update_cannonballs(cannonball_t* cpu_cannonballs, int num_cannonballs) {
  cannonball_t* gpu_cannonballs = NULL;

  // Realloc from cpu to gpu
  gpuErrchk(cudaMalloc(&gpu_cannonballs, sizeof(cannonball_t) * (num_cannonballs))); 
  gpuErrchk(cudaMemcpy(gpu_cannonballs, cpu_cannonballs, sizeof(cannonball_t) * (num_cannonballs), cudaMemcpyHostToDevice));

  int blocks = (num_cannonballs + THREADS - 1) / THREADS;

  // Calculate positions and velocities of all cannonballs in the gpu
  update_cannonballs_gpu<<<blocks, THREADS>>>(gpu_cannonballs, num_cannonballs, stars, num_stars);
  gpuErrchk(cudaDeviceSynchronize());

  // Copy udated cannonballs back to CPU
  gpuErrchk(cudaMemcpy(cpu_cannonballs, gpu_cannonballs, sizeof(cannonball_t) * num_cannonballs, cudaMemcpyDeviceToHost));

  free(gpu_cannonballs);
}

// Updates cannonballs' position and velocity concurrently using the GPU
__global__ void update_cannonballs_gpu(cannonball_t* cannonballs, int num_cannonballs, star_t* stars, int num_stars) {
  int i = (blockIdx.x * THREADS) + threadIdx.x;
  if (i < num_cannonballs) {
    cannonballs[i].x_position += cannonballs[i].x_velocity * DT;
    cannonballs[i].y_position += cannonballs[i].y_velocity * DT;

    // Loop over all stars to compute forces
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
      float combined_radius = CANNONBALL_RADIUS + stars[j].radius;
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


__host__ bool spaceship_collision(spaceship_t* spaceship, cannonball_t* cannonballs, int num_cannonballs) {
  float ship_x = spaceship->x_position;
  float ship_y = spaceship->y_position;

  // Check for collisions with all stars
  for (int i = 0; i < num_stars; i++) {
    if (check_collision(ship_x, ship_y, SPACESHIP_RADIUS, stars[i].x_position, stars[i].y_position, stars[i].radius)) {
      return true;
    }
  }
  // Check for collisions with all cannonballs
  for (int i = 0; i < num_cannonballs; i++) {
    if (check_collision(ship_x, ship_y, SPACESHIP_RADIUS,
                        cannonballs[i].x_position, cannonballs[i].y_position, CANNONBALL_RADIUS)) {
      return true;
    }
  }
  // If it hasn't found a collision thusfar, then there is none
  return false;
}
   

// Is there a collision here?
__host__ bool check_collision(float obj1_x, float obj1_y, float obj1_radius, float obj2_x, float obj2_y, float obj2_radius) {
  // Compute the distance between each obj in each dimension
  float x_diff = obj1_x - obj2_x;
  float y_diff = obj1_y - obj2_y;

  // Compute the magnitude of the distance vector
  float dist = sqrt(x_diff * x_diff + y_diff * y_diff);

  // If the distance between the objects is <= their combined radius, then there is a collision
  if (dist <= obj1_radius + obj2_radius) {
    return true;
  } else {
    return false;
  }
}














// Pay no attention to the code behind the curtain!

/***************************************\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\
\|||||||||||||||||||||||||||||||||||||||/
/|||||||||||||||||||||||||||||||||||||||\


// Old gpu lab code for posterity I guess

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
*/
