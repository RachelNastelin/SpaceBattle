Project Proposal

1. The problem: we want to play a spaceship battle game where we can shoot cannonballs at another player and avoid stars with 
gravitational pulls. There isn’t a game like that, so we want to make one. 

2. We’re going to use concurrency to quickly and efficiently update the positions of the spaceships and cannonballs. Instead
of using pthread and locks, we’re going to use kernel calls to run the threads for the GPU. We will use pthread and locks for 
other tasks. We plan to make about 6 threads, with more threads once more than 1 player is playing:
 - 1 main thread to start the game
 - 1 thread for each player to keep track of their spaceship
 - 1 thread to redraw the board
 - 1 thread that reads user input and processes controls
 - 1 thread for each cannonball that’s fired to keep track of its position
 - 1 thread for each player to communicate with the server

We’re going to use a distributed system/network with a central server so that 2 players can play the game against each other. 
We won’t use a peer-to-peer system, so every client will be connected to the server directly. We plan to handle the following
tasks on the client side:
 - Starting the game
 - Tracking the position of a player’s spaceship
 - Handling user input (including shooting cannonballs)
 - Drawing board when game starts
 - Communicating with the server
 - Comparing players’, stars’, and cannonballs’ positions
 - Sucking players and cannonballs into the stars

These will be handled on the server side:
 - Updating board based on players’ and cannonballs’ positions
 - Keeping track of the cannonballs’ positions after they’re fired
 - Communicating with the players

To play the game, the user will start the server in terminal, then in another tab of terminal, will start the game. The other 
player will then start the game.

3. Concurrency – the 6 threads described above
Networking – the distributed system/network described above

4. We plan to implement our project in roughly this order:
 - The client side, to be done in the first week (4/22 – 4/28). We will know that this is done when we can connect the client 
   to a dummy server, start the game, draw the board, move around on the board, shoot cannonballs (which won’t move after 
   being shot), and quit the game.

 - The server, to be done in the second week (4/29 – 5/5). We will know that this is done when the server can send and receive 
   messages to/from a single client, can handle client failure, generate a path for a cannonball to follow, and update the 
   board with current player and cannonball positions.

 - Connecting to the server and having multiple clients, to be done in the third week (5/6 – 5/11). We will know that this is 
   complete when we can play the game successfully on multiple computers using multiple clients. 

We will have to deal with the case where one client fails, and the other is left playing the game alone. There are likely to 
be problems with communicating players’ and cannonballs’ positions quickly enough to play the game without lag. There are also
some potential issues with closing the same thread twice and deadlock.
