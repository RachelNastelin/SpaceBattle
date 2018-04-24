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

#include "ui.h"

#define TEXT_LEN 256
#define USERNAME_LEN 8
#define SERVER_PORT 6663

 /********************************* STRUCTS **********************************/
// client join message
typedef struct server_msg {
  int clientID;
  int listen_port;
  int parent_port;
} server_msg_t;

// server response
typedef struct server_rsp {
  int clientID;
  char parent_ip[INET_ADDRSTRLEN];
  int parent_port;
} server_rsp_t;


// struct to hold our list of connections
typedef struct connect_list {
  int socket;
  struct connect_list * next;
} connect_list_t;

// struct for messages
typedef struct message {
  char username[USERNAME_LEN];
  char text[TEXT_LEN];
} message_t;


/****************************** GLOBALS **************************************/

char * server_name;
pthread_mutex_t list_lock = PTHREAD_MUTEX_INITIALIZER;
connect_list_t * connections;
int parent_socket;
int listen_port;
int parent_port;
int clientID;
bool continue_flag; // True when the client has not quit
server_msg_t * servermsg;


/*********************** FUNCTION SIGNATURES *********************************/

void * listen_relay_func (void * socket_num);

void * accept_connections_func (void * listen_socket_num);

int setup_listen ();

void remove_connection (int socket);

int socket_setup (int port, struct sockaddr_in * addr);

server_rsp_t * server_connect (server_msg_t * client_join);

void parent_connect (server_rsp_t* response, int listen_socket, bool new_thread);


/***************************** THREAD FUNCTIONS ******************************/ 

// thread to listen for and relay messages
void * listen_relay_func (void * socket_num) {
  int socket = *(int*)socket_num;
  free(socket_num);

  while (continue_flag) {
    message_t message;
    // try to read a message
    if (read(socket, &message, sizeof(message_t)) <= 0) {
      remove_connection(socket);
      // if this is the parent socket
      if (socket == parent_socket) {
        server_rsp_t * response = server_connect(servermsg);
        // if we're the new root, exit this thread
        if (response->parent_port == -1) {
          break;
        }
        // otherwise, switch to listening to the other parent
        close(socket);
        parent_connect(response, socket, false);
      } else {
        break;
      }
    } else {
      ui_add_message(message.username, message.text);
      pthread_mutex_lock(&list_lock);
      connect_list_t * current = connections;
      // relay the message to all connections
      while (current != NULL) {
        if (current->socket != socket) {
          write(current->socket, &message, sizeof(message_t));
        }
        current = current->next;
      } // while
      pthread_mutex_unlock(&list_lock);
    }
  } // while true
  close(socket);
} // listen_relay_func


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
    pthread_mutex_lock(&list_lock);
    // add the new connection to our list of connections
    connect_list_t* new_child = (connect_list_t*)malloc(sizeof(connect_list_t));
    new_child->socket = client_socket;
    new_child->next = connections;
    connections = new_child;
    pthread_mutex_unlock(&list_lock);

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


// remove a connection from our list
void remove_connection (int socket) {
  pthread_mutex_lock(&list_lock);
  connect_list_t * current = connections;
  connect_list_t * previous = NULL;
  while (current->socket != socket) {
    previous = current;
    current = current->next;
  } // while
  if (previous == NULL) {
    connections = current->next;
  } else {
    previous->next = current->next;
  } 
  free(current);
  pthread_mutex_unlock(&list_lock);
} // remove_connection


// function to set up a socket listening to a given port
int socket_setup (int port, struct sockaddr_in * addr) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s == -1) {
    perror("socket failed");
    exit(2);
  }
  // Set up addresses to connect to our parent
  addr->sin_addr.s_addr = INADDR_ANY;
  addr->sin_family = AF_INET;
  addr->sin_port = htons(port);
  return s;
} // socket_setup


// function to initialize server connection and receive a parent
server_rsp_t * server_connect(server_msg_t * client_join) {
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


// function to connect to a parent
void parent_connect(server_rsp_t* response, int listen_socket, bool new_thread){
  // set up a socket to connect to the parent
  struct sockaddr_in addr_parent;
  parent_socket = socket_setup(response->parent_port, &addr_parent);
  // set up the parent
  struct hostent* parent = gethostbyname(response->parent_ip);
  if (parent == NULL) {
    fprintf(stderr, "Unable to find host %s\n", response->parent_ip);
    exit(1);
  }
  // Specify the parent's address
  bcopy((char*)parent->h_addr, (char*)&addr_parent.sin_addr.s_addr,
        parent->h_length);
  free(response);

  // CONNECT TO PARENT
  // if the connection is unsuccessful, request a new parent
  while (connect(parent_socket, (struct sockaddr*)&addr_parent,
                 sizeof(struct sockaddr_in)) == -1) {
    // ask for a new parent
    server_rsp_t * response2 = server_connect(servermsg);
    servermsg->parent_port = response->parent_port;
    
    parent = gethostbyname(response2->parent_ip);
    if (parent == NULL) {
      fprintf(stderr, "Unable to find host %s\n", response2->parent_ip);
      exit(1);
    }
    addr_parent.sin_port = htons(response2->parent_port);
    // Specify the server's address
    bcopy((char*)parent->h_addr, (char*)&addr_parent.sin_addr.s_addr,
          parent->h_length);
    //free(client_join);
    free(response2);
  } // while

  // PREPARE TO LISTEN AND RELAY MESSAGES
  // add the parent to our list of connections
  pthread_mutex_lock(&list_lock);
  connect_list_t* parent_node = (connect_list_t*)malloc(sizeof(connect_list_t));
  parent_node->socket = parent_socket;
  parent_node->next = connections;
  connections = parent_node;
  pthread_mutex_unlock(&list_lock);

  if (new_thread) {
    // create thread to listen and relay messages from the parent
    pthread_t listen_thread;
    int * socket_num = (int*)malloc(sizeof(int));
    *socket_num = parent_socket;
    if(pthread_create(&listen_thread, NULL, listen_relay_func,
                      (void*)socket_num)) {
      perror("pthread_create failed");
      exit(EXIT_FAILURE);
    } // if
  } // if(new_thread)
} // parent_connect


/********************************** MAIN *************************************/

int main(int argc, char** argv) {
  /******************************** SET UP ***********************************/
  /******************** SET UP PART ONE: UI AND GLOBALS  *********************/
  // initialize all of our globals
  char* local_user = argv[1];
  server_name = argv[2];
  continue_flag = true;
  clientID = -1;
  servermsg = (server_msg_t*)malloc(sizeof(server_msg_t));
  
  // Initialize the chat client's user interface.
  ui_init();
  
  // Add a test message
  ui_add_message(NULL, "Type your message and hit <ENTER> to post.");

  /********* SET UP PART TWO: PREPARE TO RECEIVE CLIENT JOIN REQUESTS *******/
  // set up child socket, which will be constantly listening for incoming
  //  connections
  int listen_socket = setup_listen();
  
  /************************* CONNECT TO SERVER ******************************/
  // set up our message
  servermsg->clientID = clientID;
  servermsg->listen_port = listen_port;

  server_rsp_t * response = server_connect(servermsg);

  // edit our globals to take into account information gotten from the server
  clientID = response->clientID;
  servermsg->clientID = clientID;
  servermsg->parent_port = response->parent_port;

  /*********************** CONNECT TO PARENT *********************************/
  if (response->parent_port != -1) {
    // connect to parent and spin off a thread to listen and relay messages from
    // them
    parent_connect(response, listen_socket, true);
    parent_port = response->parent_port;
  } // if there are no previous clients, do nothing

  /********************** READ AND RELAY MESSAGES  ***************************/
  while(continue_flag) {
    // Read a message from the UI
    char* message = ui_read_input();

    // If the message is a quit command, shut down. Otherwise print the message
    if(strcmp(message, "\\quit") == 0) {
      continue_flag = false;
      servermsg->clientID = -2;
      server_connect(servermsg);
      break;
    } else if(strlen(message) > 0) {
      // Add the message to the UI
      ui_add_message(local_user, message);
      message_t * msg_struct = (message_t*)malloc(sizeof(message_t));
      strncpy(msg_struct->username, local_user, USERNAME_LEN);
      strncpy(msg_struct->text, message, TEXT_LEN);
      connect_list_t * current = connections;
      while (current != NULL) {
        write(current->socket, msg_struct, sizeof(message_t));
        current = current->next;
      }
      free(msg_struct);
    }

    // Free the message
    free(message);
  }

  // free our server message and close the listening_socket
  free(servermsg);  
  close(listen_socket);
   
  // Clean up the UI
  ui_shutdown(); 

} // main
