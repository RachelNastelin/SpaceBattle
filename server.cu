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
typedef struct talk_to_client_args {
  int clientID;
  int port;
  int socket;
  spaceship_t * ship;
  int direction; //TODO: initialize this somewhere
  bool cannonball_shot; //TODO: initialize this somewhere
} talk_to_client_args_t;

/****************************** GLOBALS **************************************/

// GLOBAL CLIENT LIST
client_list_t * clients;
int client_count;

server_rsp_t * send_to_clients;
pthread_mutex_t send_to_clients_lock;

cannonball_t * cannonballs;
pthread_mutex_t cannonballs_lock;
int num_cannonballs;

/**************************** FUNCTIONS ***************************************/
/*************************** SIGNATURES ***************************************/
void stop_game();
void remove_client (int port);
void quit_client (int port);
void end_game ();

/*************************** THREAD FUNCTIONS *********************************/
void * talk_to_client(void * args){
  talk_to_client_args_t * client_info = (talk_to_client_args_t *)args;
  //client_info->clientID = client_count;

  
  while(true){
    // make sure that all the clients are still connected
    for(int i = 0; i < 2; i++){
      if(clients[i].socket == LOST_CONNECTION){
        remove_client(client_info->port);
      } // if
    } // for
    
    // listen for information from client
    msg_to_server * response = (msg_to_server*)malloc(sizeof(msg_to_server_t));
    read(client_info->socket, response, sizeof(msg_to_server_t));

    // call functions to handle information
    if(client_info->cannonball_shot){
      if(cannonballs == NULL){
	cannonballs = (cannonball_t*)malloc(sizeof(cannonball_t));
      } // if
      pthread_mutex_lock(&cannonballs_lock);
      add_cannonball(cannonballs, num_cannonballs);
      pthread_mutex_unlock(&cannonballs_lock);
      num_cannonballs++;
    } // if a cannonball was shot
    
    // put information together with information about other client
    // step 1: which client are we working with?
    int i = 0;
    while(clients[i].clientID != client_info->clientID);
    if (i ==0){
      // step 2: change the info in send_to_clients for the client you're
      //         working with
      pthread_mutex_lock(&send_to_clients_lock);
      send_to_clients->clientID0 = clients[i].clientID;
      send_to_clients->client_socket0 = clients[i].socket;
      send_to_clients->listen_port0 = clients[i].port_num;
      send_to_clients->ship0 = update_spaceship(client_info->ship,
						client_info->direction);
      pthread_mutex_unlock(&send_to_clients_lock);
    } else if (i == 1){
      // step 2: change the info in send_to_clients for the client you're
      //         working with
      pthread_mutex_lock(&send_to_clients_lock);
      send_to_clients->clientID1 = clients[i].clientID;
      send_to_clients->client_socket1 = clients[i].socket;
      send_to_clients->listen_port1 = clients[i].port_num;
      send_to_clients->ship1 = update_spaceship(client_info->ship,
						client_info->direction);
      pthread_mutex_unlock(&send_to_clients_lock);
    }

    // send information about both clients
    for(int j = 0; j < 2; j++){
      send_to_clients->target_clientID = j;
      write(clients[i].socket, send_to_clients, sizeof(server_rsp_t));
    } // for
  } // while
} // talk_to_client
/************************* END GAME FUNCTIONS *********************************/
void stop_game(){
  server_rsp_t quit_msg;
  for(int i = 0; i < 2; i++){
    if(i == 0){
      quit_msg.clientID0 = clients[i].clientID;
      quit_msg.continue_flag = false; // stops client threads
      quit_msg.listen_port0 = clients[i].port_num;
    }
    else{
      quit_msg.clientID1 = clients[i].clientID;
      quit_msg.continue_flag = false; // stops client threads
      quit_msg.listen_port1 = clients[i].port_num;
    }
    write(clients[i].socket, &quit_msg, sizeof(server_rsp_t));
  }
  free(clients);
  free(send_to_clients);
  pthread_mutex_destroy(&send_to_clients_lock);
  free(cannonballs);
  pthread_mutex_destroy(&cannonballs_lock);

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
  /*================ SET UP: PART 1, SET UP SERVER SOCKET ====================*/
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
  

  /*=================== SET UP: PART 2, SET UP GLOBALS ======================*/
  client_count = 0;
  // set up the list of connected clients
  clients = (client_list_t*)malloc(sizeof(client_list_t));
  int client_socket;
  pthread_mutex_init(&(send_to_clients_lock), NULL);
  pthread_mutex_init(&(cannonballs_lock), NULL);

  
  /*====================== ACCEPT CLIENT CONNECTIONS ========================*/
  // Accept 2 connections
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

    /* STORE SOCKET AND SHIP FOR NEW CLIENT */
    clients[client_count - 1].socket = client_socket;
    
    spaceship_t * ship = (spaceship_t*)malloc(sizeof(spaceship_t));
    clients[client_count - 1].ship = init_spaceship(ship, client_count);
    
    /* LISTEN TO CLIENT */
    msg_to_server_t message;
    if(read(client_socket, &message, sizeof(server_rsp_t)) == -1){
      // if the server couldn't read from the client, exit the game
      remove_client(message.listen_port);
    }

    /* STORE OTHER INFORMATION ABOUT NEW CLIENT */
    // store the client's ip address 
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ipstr, INET_ADDRSTRLEN);
    
    client_list_t* new_client = (client_list_t*)malloc(sizeof(client_list_t));
    new_client->clientID = client_count;
    strncpy(new_client->ip, ipstr, INET_ADDRSTRLEN);
    new_client->port_num = message.listen_port;
    new_client->next = clients;
    clients = new_client;
    client_count++;

    
    /*============= SET UP COMMUNICATION WITH NEW CLIENT ====================*/ 
    // make new thread to communicate with client
    pthread_t new_client_thread;
    talk_to_client_args_t * args = (talk_to_client_args_t*)
      malloc(sizeof(talk_to_client_args_t));
    args->port = new_client->port_num;
    args->socket = client_socket;
    args->clientID = new_client->clientID;
    args->ship = clients[client_count - 1].ship; 
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
