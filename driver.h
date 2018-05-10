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
  /* Client 0 */
  int client_socket0;
  int clientID0;
  int listen_port0;
  spaceship_t * ship0;
  
  /* Client 1 */
  int client_socket1;
  int clientID1;
  int listen_port1;
  spaceship_t * ship1;

  /* Same for both clients  */
  cannonball_t * cannonballs; // array used for determining spaceship death
  bool continue_flag; // when false, stops all threads on client side
  int target_clientID;
  
} server_rsp_t;

// client storage for the server's internal list of clients.
// each client_list variable represents one client in the list.
typedef struct client_list {
  int clientID;
  char ip[INET_ADDRSTRLEN]; // IP address of client
  int port_num; // port of client
  int socket;
  spaceship_t * ship;
  struct client_list * next;
} client_list_t;

__host__ star_t* init_stars();
__host__ spaceship_t* init_spaceship(spaceship_t* spaceship, int clientID);
__host__ cannonball_t* init_cannonball(spaceship_t* spaceship, int direction_shot);
__host__ bool cannonball_in_bounds(cannonball_t* cannonball);
__host__ void add_cannonball(cannonball_t* cannonballs, int num_cannonballs);
__host__ spaceship_t* update_spaceship(spaceship_t* spaceship, int direction_boost);
__host__ cannonball_t*  update_cannonballs(cannonball_t* cannonballs, int num_cannonballs);
__global__ void update_cannonballs_gpu(cannonball_t* cannonballs, int num_cannonballs);
__host__ bool spaceship_collision(spaceship_t* spaceship, cannonball_t* cannonballs, int num_cannonballs);
__host__ bool check_collision(float obj1_x, float obj1_y, float obj1_radius, float obj2_x, float obj2_y, float obj2_radius);

#endif
