#ifndef __DRIVER_H__
#define __DRIVER_H__

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
typedef struct star {
  float mass;
  float radius;
  float x_position;
  float y_position;
} star_t;

__host__ star_t* create_stars();
__host__ cannonball_t* add_cannonball(spaceship_t* spaceship, cannonball_t* cannonballs, int num_cannonballs, int direction_shot);
__host__ spaceship_t* update_spaceship(spaceship_t* spaceship, star_t* stars, int direction_boost);
__host__ cannonball_t*  update_cannonballs(cannonball_t* cannonballs, star_t* stars, int num_cannonballs, int num_stars);
__global__ void update_cannonballs_gpu(cannonball_t* cannonballs, star_t* stars, int num_cannonballs, int num_stars);
bool check_for_collision(spaceship_t* spaceship, cannonball_t* cannonball, star_t* star);
bool within_bounds(int ship_pos, int obstacle_pos, int obstacle_radius);

#endif
