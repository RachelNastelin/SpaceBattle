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

#include "board.h"

/**
 * Initialize the board display by printing the title and edges
 */
void init_display(){}

/**
 * Show a game over message and wait for a key press.
 */
void end_game(){}

/**
 * Run in a thread to draw the current state of the game board.
 */
void draw_board(){}

/**
 * Run in a thread to move the ship around on the board
 */
void update_ship(){}

/**
 * Run in a thread to generate cannonballs on the board.
 */
void generate_cannonball(){}
