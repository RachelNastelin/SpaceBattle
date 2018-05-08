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
#include <time.h>
#include <cuda.h>

#include "board.h"
#include "driver.h"

#define LOST_CONNECTION -2 // clients that the server lost its connection with
#define SERVER_PORT 6664
// directions, used for user input
#define UP 1
#define DOWN 2
#define RIGHT 3
#define LEFT 4
/********************************* STRUCTS **********************************/

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

typedef struct talk_to_client_args {
  int port;
  int socket;
} talk_to_client_args_t;

/****************************** GLOBALS **************************************/

// GLOBAL CLIENT LIST
client_list_t * clients;
int client_count;

/**************************** FUNCTIONS **************************************/
/*************************** SIGNATURES ***************************************/
void stop_game();
void remove_client (int port);
void quit_client (int port);
void end_game ();

/*************************** THREAD FUNCTIONS *********************************/
void * talk_to_client(void * args){
  talk_to_client_args_t * client_info = (talk_to_client_args_t *)args;
  while(true){
     // make sure that all the clients are still connected
    for(int i = 0; i < 2; i++){
      if(clients[i].socket == LOST_CONNECTION){
        remove_client(client_info->port);
      } // if
    } // for
    
    // listen for information
    msg_to_server * response = (msg_to_server*)malloc(sizeof(msg_to_server_t));
    read(client_info->socket, response, sizeof(msg_to_server_t));

    // call functions to handle information
    
    // put information together with information about other client
    // send information about both clients
  } // while
} // talk_to_client
/************************* END GAME FUNCTIONS *********************************/
void stop_game(){
  server_rsp_t quit_msg;
  for(int i = 0; i < 2; i++){
    quit_msg.clientID = clients[i].clientID;
    quit_msg.continue_flag = false; // stops client threads
    quit_msg.listen_port = clients[i].port_num;
    // send quit_msg
    write(clients[i].socket, &quit_msg, sizeof(server_rsp_t));
  }
  free(clients);
  exit(1);
}

// called when a client cannot be communicated with.
void remove_client (int port) {
  // TODO: change this to whatever print function we're using for the UI
  printf("Your opponent can't connect to the server.\n");
  stop_game();
}

// called when a client quits before the game finishes
void quit_client (int port){
  // TODO: change this to whatever print function we're using for the UI
  printf("One player has quit the game.\n");
  //server_rsp_t quit_msg;
  stop_game();
} // quit_client

// called when the game ends, which is when at least one player has died
void end_game (){
  // TODO: announce winner
  //server_rsp_t quit_msg;
  stop_game();
} // end_game

/***************************** MAIN *******************************************/

int main() {
  // Set up a socket
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if(s == -1) {
    perror("socket");
    exit(2);
  }

  // Listen at this address.
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(SERVER_PORT);
  

  // Bind to the specified address
  if(bind(s, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    perror("bind");
    exit(2);
  }

  // Become a server socket
  if(listen(s, 2)) {
    perror("listen failed");
    exit(2);
  }

  client_count = 0;
  // set up the list of connected clients
  clients = (client_list_t*)malloc(sizeof(client_list_t));
  int client_socket;
  server_rsp_t * response;

  
  // Repeatedly accept connections
  while(client_count <= 2) {
    // Accept a client connection
    struct sockaddr_in client_addr;
    socklen_t client_addr_length = sizeof(struct sockaddr_in);
    client_socket = accept(s, (struct sockaddr*)&client_addr,
                           &client_addr_length);
    
    if(client_socket == -1) {
      perror("accept failed");
      exit(2);
    }
    clients[client_count - 1].socket = client_socket;
    // read a message from the client
    msg_to_server_t message;
    if(read(client_socket, &message, sizeof(server_rsp_t)) == -1){
      // if the server couldn't read from the client, exit the game
      remove_client(message.listen_port);
    }

    printf("\n/-----------------------------------------------------------/\n");

    // store the client's ip address 
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ipstr, INET_ADDRSTRLEN);
   
    // set up the response to the client
    response = (server_rsp_t*)malloc(sizeof(server_rsp_t));

    // add the new client to our list of clients
    client_list_t* new_client = (client_list_t*)malloc(sizeof(client_list_t));
    new_client->clientID = client_count;
    strncpy(new_client->ip, ipstr, INET_ADDRSTRLEN);
    new_client->port_num = message.listen_port;
    new_client->next = clients;
    clients = new_client;
    client_count++;

    // make new thread to communicate with client
    pthread_t new_client_thread;
    talk_to_client_args_t * args = (talk_to_client_args_t*)malloc(
                                        sizeof(talk_to_client_args_t));
    args->port = new_client->port_num;
    args->socket = client_socket;
    pthread_create(&new_client_thread, NULL, talk_to_client, (void *)(args));
    
    // end game if necessary
    if (message.continue_flag == false) {
      quit_client(message.listen_port);
    } else if(message.died == true){
      end_game();
    } else {
      // if they aren't trying to quit, connect them
      printf("\nClient %d connected from %s, on port %d\n",
             message.clientID, ipstr, ntohs(message.listen_port));
    } // else
  } // while
  
  // respond to the client
  write(client_socket, response, sizeof(server_rsp_t));
  // close the socket
  close(client_socket);

  // print the current client list
  printf("\nCURRENT CLIENT LIST:\n");
  client_list_t* current = clients;
  while (current != NULL) {
    printf("Client %d, at %s, on port %d.\n", current->clientID, current->ip,
           current->port_num);
    current = current->next;
  }
  
  close(s);
}
