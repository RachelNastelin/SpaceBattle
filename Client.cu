#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdbool.h>
#include <cuda.h>

#include "board.h"
#include "driver.h"

#define SERVER_PORT 6664
#define NOT_IN_USE -1 // sockets not in use have this value
#define LOST_CONNECTION -2 // clients that the server lost its connection with
// directions, used for user input
#define UP 1
#define DOWN -1
#define RIGHT 2
#define LEFT -2
/********************************* STRUCTS **********************************/
typedef struct user_input {
  int direction; // LEFT, RIGHT, UP, or DOWN
} user_input_t;

/****************************** GLOBALS **************************************/

char * server_name;
int connections[2]; // Each index has a socket number
int num_connections;
int global_listen_port;
int global_clientID;
bool global_continue_flag; // True when the client has not quit
pthread_mutex_t connections_lock = PTHREAD_MUTEX_INITIALIZER;
//SDL_Renderer* renderer = NULL;

/*********************** FUNCTION SIGNATURES *********************************/

void * listen_relay_func (void * socket_num);
void remove_connection (int index);
int socket_setup (int port, struct sockaddr_in * addr);

/***************************** THREAD FUNCTIONS ******************************/

// thread to accept connections
void * accept_connections_func (void * listen_socket_num) {
  int socket = *(int*)listen_socket_num;
  free(listen_socket_num);

  // Repeatedly accept connections
  while(global_continue_flag) {
    // Accept a client connection
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(struct sockaddr_in);
    int client_socket = accept(socket, (struct sockaddr*)&client_addr,
                               &client_addr_len);
    pthread_mutex_lock(&connections_lock);
    // add the new connection to our list of connections
    connections[num_connections-1] = client_socket;
    num_connections++;
    pthread_mutex_unlock(&connections_lock);
    
    // create a listen and relay thread for the new connection
    pthread_t listen_thread;
    int * socket_num = (int*)malloc(sizeof(int));
    *socket_num = client_socket;
    if(pthread_create(&listen_thread, NULL, listen_relay_func,
                      (void*)socket_num)) {
      perror("pthread_create failed");
      exit(EXIT_FAILURE);
    } // if  
  } // while
  return NULL;
} // accept_connections_func

// thread to listen for and relay messages
void * listen_relay_func (void * socket_num) {
  int socket = *(int*)socket_num;
  free(socket_num);

  while (global_continue_flag) {
   
    server_rsp_t message;
    // try to read a message
    
    if (read(socket, &message, sizeof(msg_to_server_t)) <= 0) {
      // something went wrong, exit
      remove_connection(socket);
      
    } else {
      // the information was sent successfully
      global_continue_flag = message.continue_flag; // stops threads on client side
      
      // update your game board
      // do some stuff
    }
  } // while true
  close(socket);
  return NULL;
} // listen_relay_func

/*************************** HELPER FUNCTIONS ********************************/
// remove a connection from our list
// MAKE THIS END THE PROGRAM
void remove_connection (int index) {
  pthread_mutex_lock(&connections_lock);
  connections[index] = LOST_CONNECTION;
  pthread_mutex_unlock(&connections_lock);
} // remove_connection


// setup a socket to listen for connections (and spin off a listening thread)
int setup_listen() {
  // set up child socket, which will be constantly listening for incoming
  //  connections
  struct sockaddr_in addr_listen;
  int listen_socket = socket_setup(0, &addr_listen);

  // Bind to the specified address
  if(bind(listen_socket, (struct sockaddr*)&addr_listen,
          sizeof(struct sockaddr_in))) {
    perror("bind");
    exit(2);
  }
  // Start listening on this socket
  if(listen(listen_socket, 2)) {
    perror("listen failed");
    exit(2);
  }
  // Get the listening socket info so we can find out which port we're using
  socklen_t addr_size = sizeof(struct sockaddr_in);
  getsockname(listen_socket, (struct sockaddr *) &addr_listen, &addr_size);
  
  // save the port we're listening on
  global_listen_port = ntohs(addr_listen.sin_port);

  // Spin up a thread to constantly listen for connections on this socket
  pthread_t accept_connections;
  int * listen_socket_num = (int*)malloc(sizeof(int));
  *listen_socket_num = listen_socket;
  if(pthread_create(&accept_connections, NULL, accept_connections_func,
                    (void*)listen_socket_num)) {
    perror("pthread_create failed");
    exit(EXIT_FAILURE);
  }
  return listen_socket;
} // setup_listen

// function to initialize server connection and receive a parent
server_rsp_t * server_connect(msg_to_server_t * client_join) {
  // set up socket to connect to server
  struct sockaddr_in addr;
  int s = socket_setup(SERVER_PORT, &addr);
  // set up the server as passed into the command line
  struct hostent* server = gethostbyname(server_name);
  if (server == NULL) {
    fprintf(stderr, "Unable to find host %s\n", server_name);
    exit(1);
  }
  // Specify the server's address
  bcopy((char*)server->h_addr, (char*)&addr.sin_addr.s_addr, server->h_length);
  // Connect to the server
  if(connect(s, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    perror("connect failed");
    exit(2);
  } // if

  // send client join message (send the port that the client is listening on)
  write(s, client_join, sizeof(server_rsp_t));

  server_rsp_t * response = (server_rsp_t*)malloc(sizeof(server_rsp_t));
  read(s, response, sizeof(server_rsp_t));

  close(s);
  // return server's response
  return response;
} // server_connect



// function to set up a socket listening to a given port
int socket_setup (int port, struct sockaddr_in * addr) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s == -1) {
    perror("socket failed");
    exit(2);
  }
  // Set up addresses
  addr->sin_addr.s_addr = INADDR_ANY;
  addr->sin_family = AF_INET;
  addr->sin_port = htons(port);
  return s;
} // socket_setup

/********************************** MAIN *************************************/
int main(int argc, char**argv){
  /******************** SET UP PART ONE: UI AND GLOBALS  *********************/
  server_name = argv[1];
  global_continue_flag = true;
  msg_to_server_t * msg_to_server = (msg_to_server_t*)malloc(sizeof(msg_to_server_t));
  //global_listen_port =
  
  // set up connections array
  for(int i = 0; i < 3; i++){
    connections[i] = NOT_IN_USE;
  } // for

  gui_init();
  //star_t * stars = init_stars();
  //color_t star_color = {0,0,0,255};
  //gui_draw_star(stars[0].x_position, stars[0].y_position, stars[0].radius, star_color);
  // gui_draw_star(stars[1].x_position, stars[1].y_position, stars[1].radius, star_color);

  /********* SET UP PART TWO: PREPARE TO RECEIVE CLIENT JOIN REQUESTS *******/
  // set up child socket, which will be constantly listening for incoming
  //  connections
  int listen_socket = setup_listen();

  /************************* CONNECT TO SERVER ******************************/
  msg_to_server->clientID = global_clientID;
  msg_to_server->listen_port = global_listen_port; //updated in setup_listen
  msg_to_server->continue_flag = global_continue_flag;
  server_rsp_t * response = server_connect(msg_to_server);

  // edit our globals to take into account information gotten from the server
  if(response->target_clientID == 0){
    global_clientID = response->clientID0;
  }
  else{
    global_clientID = response->clientID1;

  }
  msg_to_server->clientID = global_clientID;
  

  /************************* DISPLAY BOARD **********************************/
  gui_draw_ship(response->ship0->x_position,response->ship0->y_position);
  gui_draw_ship(response->ship1->x_position,response->ship1->y_position);

  //SDL_RenderDrawLine(renderer, 320, 200, 300, 240);
  //SDL_RenderDrawLine(renderer, 300, 240, 340, 240);
                

  // End
  free(msg_to_server);
  free(response);
  //free(stars);
  close(listen_socket);
} // main
