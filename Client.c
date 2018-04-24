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

#define SERVER_PORT 6663
#define NOT_IN_USE -1 // sockets not in use have this value

/********************************* STRUCTS **********************************/

typedef struct update_msg{
  int clientID; // to differentiate between players
  int player_pos[][];
  bool quitting; // 0 = not quitting, 1 = quitting
  bool cannonball_shot; // 0 = didn't shoot cannonball, 1 = shot cannonball
  int direction; // LEFT, RIGHT, UP, or DOWN
} update_msg_t;


/****************************** GLOBALS **************************************/

char * server_name;
int connections[2]; // Each index has a socket number
int num_clients;
pthread_mutex_t connections_lock = PTHREAD_MUTEX_INITIALIZER;

/***************************** THREAD FUNCTIONS ******************************/

// thread to accept connections
void * accept_connections_func (void * listen_socket_num) {
  int socket = *(int*)listen_socket_num;
  free(listen_socket_num);

  // Repeatedly accept connections
  while(continue_flag) {
    // Accept a client connection
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(struct sockaddr_in);
    int client_socket = accept(socket, (struct sockaddr*)&client_addr,
                               &client_addr_len);
    pthread_mutex_lock(&connections_lock);
    // add the new connection to our list of connections
    connect_list_t* new_child = (connect_list_t*)malloc(sizeof(connect_list_t));
    new_child->socket = client_socket;
    new_child->next = connections;
    connections = new_child;
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
} // accept_connections_func


/*************************** HELPER FUNCTIONS ********************************/
// remove a connection from our list
void remove_connection (int index) {
  pthread_mutex_lock(&connections_lock);
  connections[index] = NOT_IN_USE;
  pthread_mutex_unlock(&connections_lock);
} // remove_connection

/********************************** MAIN *************************************/
int main(int argc, char**argv){
/******************** SET UP PART ONE: UI AND GLOBALS  *********************/
  char* local_user = argv[1];
  server_name = argv[2];
  
  // set up connections array
  for(int i = 0; i < 3; i++){
    connections[i] = NOT_IN_USE;
  } // for

/********* SET UP PART TWO: PREPARE TO RECEIVE CLIENT JOIN REQUESTS *******/
// set up child socket, which will be constantly listening for incoming
//  connections
  int listen_socket = setup_listen();
  
} // main
