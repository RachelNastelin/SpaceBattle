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
  int listen_port; 
  bool quitting; // 0 = not quitting, 1 = quitting
  bool cannonball_shot; // 0 = didn't shoot cannonball, 1 = shot cannonball
  int direction; // LEFT, RIGHT, UP, or DOWN
} update_msg_t;

// server response
typedef struct server_rsp {
  int clientID;
} server_rsp_t;


/****************************** GLOBALS **************************************/

char * server_name;
int connections[2]; // Each index has a socket number
int num_connections;
int global_listen_port;
int global_clientID;
bool continue_flag; // True when the client has not quit
pthread_mutex_t connections_lock = PTHREAD_MUTEX_INITIALIZER;

/*********************** FUNCTION SIGNATURES *********************************/
void listen_relay_func (void * socket_num);


/***************************** THREAD FUNCTIONS ******************************/

// thread to accept connections
void accept_connections_func (void * listen_socket_num) {
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
    connections[num_connections-1] = client_socket;
    num_connections++;

    /*
    connect_list_t* new_child = (connect_list_t*)malloc(sizeof(connect_list_t));
    new_child->socket = client_socket;
    new_child->next = connections;
    connections = new_child;
    */
    
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

// thread to listen for and relay messages
void listen_relay_func (void * socket_num) {
  int socket = *(int*)socket_num;
  free(socket_num);

  while (continue_flag) {
    message_t message;
    // try to read a message
    if (read(socket, &message, sizeof(message_t)) <= 0) {
      // something went wrong, exit
      remove_connection(socket);
    } else {
      ui_add_message(message.username, message.text);
      pthread_mutex_lock(&list_lock);
      int ** current = connections;
      // relay the message to all connections
      while (current != NULL) {
        if () {
          write(current->socket, &message, sizeof(message_t));
        }
        current = current->next;
      } // while
      pthread_mutex_unlock(&list_lock);
    }
  } // while true
  close(socket);
} // listen_relay_func


/*************************** HELPER FUNCTIONS ********************************/
// remove a connection from our list
void remove_connection (int index) {
  pthread_mutex_lock(&connections_lock);
  connections[index] = NOT_IN_USE;
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
  listen_port = ntohs(addr_listen.sin_port);

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
void server_rsp_t * server_connect(server_msg_t * client_join) {
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
  write(s, client_join, sizeof(server_msg_t));

  server_rsp_t * response = (server_rsp_t*)malloc(sizeof(server_rsp_t));
  read(s, response, sizeof(server_rsp_t));

  close(s);
  // return server's response
  return response;
} // server_connect



/********************************** MAIN *************************************/
int main(int argc, char**argv){
/******************** SET UP PART ONE: UI AND GLOBALS  *********************/
  server_name = argv[1];
  continue_flag = true;
  update_msg_t * msg_to_server = (update_msg_t*)malloc(sizeof(update_msg_t));
  global_clientID = -1;
  //global_listen_port = 
  
  // set up connections array
  for(int i = 0; i < 3; i++){
    connections[i] = NOT_IN_USE;
  } // for

  //creates threads here
  
  /********* SET UP PART TWO: PREPARE TO RECEIVE CLIENT JOIN REQUESTS *******/
  // set up child socket, which will be constantly listening for incoming
  //  connections
  int listen_socket = setup_listen();

  /************************* CONNECT TO SERVER ******************************/
  msg_to_server->clientID = global_clientID;
  msg_to_server->listen_port = global_listen_port; //updated in setup_listen

  
  
  
} // main
