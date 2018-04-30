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

#define SERVER_PORT 6663

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
  int listen_port;
  /* Add things */
} server_rsp_t;

// client storage for the directory server's internal list of clients
typedef struct client_list {
  int clientID;
  char ip[INET_ADDRSTRLEN];
  int port_num;
  struct client_list * next;
} client_list_t;

/****************************** GLOBALS **************************************/

// GLOBAL CLIENT LIST
client_list_t * clients;
int client_count;

/*********************** FUNCTIONS *******************************************/

// remove a client from the list, return the ID of the removed client
// (or -1 if the client is not in the list)
int remove_client (int port) {
  /* Shut down everything */
  return -1;
}

/***************************** MAIN *******************************************/

int main() {
  // Set up a socket
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if(s == -1) {
    perror("socket");
    exit(2);
  }

  // Listen at this address.
  struct sockaddr_in addr = {
    .sin_addr.s_addr = INADDR_ANY,
    .sin_family = AF_INET,
    .sin_port = htons(SERVER_PORT)
  };

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
  clients = NULL;
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
    
    // read a message from the client
    server_rsp_t message;
    read(client_socket, &message, sizeof(server_rsp_t));

    printf("\n/-----------------------------------------------------------/\n");

    // store the client's ip address 
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ipstr, INET_ADDRSTRLEN);
   
    //set up the response to the client
    response = (server_rsp_t*)malloc(sizeof(server_rsp_t));

      // Add the new client to our list of clients
      client_list_t* new_client = (client_list_t*)malloc(sizeof(client_list_t));
      new_client->clientID = client_count;
      strncpy(new_client->ip, ipstr, INET_ADDRSTRLEN);
      new_client->port_num = message.listen_port;
      new_client->next = clients;
      clients = new_client;
      client_count++;

      // if the client's ID is -2, then the client is quitting 
      if (message.clientID == -2) {
        printf("Client %d is exiting.\n", remove_client(message.listen_port));
      } else {
        printf("\nClient %d connected from %s, on port %d\n",
               message.clientID, ipstr, ntohs(message.listen_port));
      } // else
    }

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
