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

// client storage for the directory server's internal list of clients
typedef struct client_list {
  int clientID;
  char ip[INET_ADDRSTRLEN];
  int port_num;
  struct client_list * next;
} client_list_t;

// GLOBAL CLIENT LIST
client_list_t * clients;
int client_count;

// remove a client from the list, return the ID of the removed client
// (or -1 if the client is not in the list)
int remove_client (int port) {
  client_list_t * current = clients;
  client_list_t * previous = NULL;
  while (current != NULL) {
    if (current->port_num == port) {
      if (previous == NULL) {
        clients = current->next;
      } else {
        previous->next = current->next;
      }
      client_count--;
      int parentID = current->clientID;
      free(current);
      return parentID;
    }
    previous = current;
    current = current->next;
  }
  return -1;
}


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

  // Repeatedly accept connections
  while(client_count <= 2) {
    // Accept a client connection
    struct sockaddr_in client_addr;
    socklen_t client_addr_length = sizeof(struct sockaddr_in);
    int client_socket = accept(s, (struct sockaddr*)&client_addr,
                               &client_addr_length);

    if(client_socket == -1) {
      perror("accept failed");
      exit(2);
    }
    
    // read a message from the client
    server_msg_t message;
    read(client_socket, &message, sizeof(server_msg_t));

    printf("\n/-----------------------------------------------------------/\n");

    // store the client's ip address 
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ipstr, INET_ADDRSTRLEN);
   
    //set up the response to the client
    server_rsp_t * response = (server_rsp_t*)malloc(sizeof(server_rsp_t));

    // if the client's ID is -1, then it is the client's first connection
    if (message.clientID == -1) {
      printf("Client's first connection from %s, on port %d!\nNew ID is: %d.\n",
             ipstr, ntohs(message.listen_port), client_count);
      response->clientID = client_count;
      find_parent(client_count, response, clients);

      // Add the new client to our list of clients
      client_list_t* new_client = (client_list_t*)malloc(sizeof(client_list_t));
      new_client->clientID = client_count;
      strncpy(new_client->ip, ipstr, INET_ADDRSTRLEN);
      new_client->port_num = message.listen_port;
      new_client->next = clients;
      clients = new_client;
      client_count++;

      // if the client's ID is not -1, then the client is either quitting
      // or requesting a new parent
    } else {
      if (message.clientID == -2) {
        printf("Client %d is exiting.\n", remove_client(message.listen_port));
      } else {
        printf("\nClient %d connected from %s, on port %d\n",
               message.clientID, ipstr, ntohs(message.listen_port));
        printf("Client %d needs a new parent.\n", message.clientID);
        remove_client(message.parent_port);

        // find a valid parent
        int current_index = 1;
        client_list_t * current = clients;
        while (current->clientID != message.clientID) {
          current_index++;
          current = current->next;
          if (current == NULL) {
            break;
          }
        }
        current = current->next;
        int clients_remaining = client_count - current_index;
        if (clients_remaining == 0) {
          printf("Client is new root.\n");
          response->parent_port = -1;
        } else {
          find_parent(clients_remaining, response, current);
        }
      }
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
  }
  close(s);
}
