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

// information sent from client to server
typedef struct msg_to_server{
  int clientID; // to differentiate between players
  int listen_port;
  bool died; // true if the spaceship intersected with a cannonball or star
  bool quitting; // 0 = not quitting, 1 = quitting
  bool cannonball_shot; // 0 = didn't shoot cannonball, 1 = shot cannonball
  int direction; // LEFT, RIGHT, UP, or DOWN
  bool continue_flag; // when false, stops all threads on client side 
} msg_to_server_t;

// information sent from server to client
typedef struct server_rsp {
  int client_socket;
  int clientID;
  int listen_port;
  cannonball_t * cannonballs; // array used for determining spaceship death
  bool continue_flag; // when false, stops all threads on client side
  
  // TODO: the board, however we're storing it
  /* Add things */
} server_rsp_t;

// client storage for the server's internal list of clients.
// each client_list variable represents one client in the list.
typedef struct client_list {
  int clientID;
  char ip[INET_ADDRSTRLEN]; // IP address of client
  int port_num; // port of client
  int socket;
  struct client_list * next; 
} client_list_t;

__host__ star_t* create_stars();
__host__ cannonball_t* add_cannonball(spaceship_t* spaceship, cannonball_t* cannonballs, int num_cannonballs, int direction_shot);
__host__ spaceship_t* update_spaceship(spaceship_t* spaceship, star_t* stars, int direction_boost);
__host__ cannonball_t*  update_cannonballs(cannonball_t* cannonballs, star_t* stars, int num_cannonballs, int num_stars);
__global__ void update_cannonballs_gpu(cannonball_t* cannonballs, star_t* stars, int num_cannonballs, int num_stars);
bool check_for_collision(spaceship_t* spaceship, cannonball_t* cannonball, star_t* star);
bool within_bounds(int ship_pos, int obstacle_pos, int obstacle_radius);

#endif
