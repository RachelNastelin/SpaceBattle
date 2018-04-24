Distributed Systems Lab
=======================

## How to test it
### Example Shell Commands
Make a server in one terminal window with the command:
```
server
```
In another terminal tab, make a client with the command:
```
client <username> localhost
```
You can do this as many times as you wish, and you'll notice that the server will print information on the clients. You'll be able to see:
* The list of all clients, including their IDs and their ports
* Which clients connect to which parents
* When a new client connects:
  * Which IP it's connecting from
  * Which port it's using
* When a client loses connection to its parent:
  * Which parent it had been connected to
  * Which client becomes its new parent
  * If the client becomes the root, this will also be printed
These prints will help you test the program. 
## What doesn't work
If the directory server goes down while it still has clients, clients can no longer quit. If those clients try to find new parents, the new server won't have those clients in its list, so there will be a segmentation fault. Cannot recover from server failure in general.
## Example chat
In one tab:
```
server
```
In a second tab:
```
client Betty localhost
```
In a third tab:
```
client Barnaby localhost
```
In second tab:
```
hello
```
In third tab:
```
Hi Betty
```
In second tab:
```
Ew I want to talk to Brendan, not you.
```
```
\quit
```
In third tab:
```
\quit
```
## What you should expect if there are bugs or inconsistent results
### bind: Address already in use
 If it prints 
 ```
 bind: Address already in use
 ```
 when you start the server, then go into clientv1.c and change SERVER_PORT to some other valid port number. Change SERVER_PORT in serverv1.c so it matches SERVER_PORT in clientv1.c. This just happens because we don't have a way for the server to exit cleanly, so it doesn't let go of ports when it's done with them.
