
gCC = clang
CFLAGS = -g

all: client server

clean:
	rm -f client
	rm -f server

client: Client.c
	$(CC) $(CFLAGS) -o client Client.c -lncurses -lpthread
server: server.c
	$(CC) $(CFLAGS) -o server server.c -lncurses -lpthread
