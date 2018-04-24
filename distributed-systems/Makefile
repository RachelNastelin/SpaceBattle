gCC = clang
CFLAGS = -g

all: client server

clean:
	rm -f client
	rm -f server

client: client.c ui.c ui.h
	$(CC) $(CFLAGS) -o client client.c ui.c -lncurses -lpthread

server: server.c
	$(CC) $(CFLAGS) -o server server.c -lncurses -lpthread
