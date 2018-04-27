
gCC = clang
CFLAGS = -g

all: client

clean:
	rm -f client

client: Client.c
	$(CC) $(CFLAGS) -o client Client.c -lncurses -lpthread
