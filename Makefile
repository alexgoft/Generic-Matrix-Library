
DEPENDENCIES = Matrix.hpp Complex.h MatrixException.h

FLAGS = -std=c++11 -Wextra -Wall -pthread -Wvla -O -DNDEBUG -c

CC = g++

Matrix: $(DEPENDENCIES)
		$(CC) $(FLAGS) Matrix.hpp -o Matrix.hpp.gch

tar: 
	tar cvf ex3.tar Makefile Matrix.hpp MatrixException.h README

clean:
	rm -f Matrix.hpp.gch *.O ex3.tar

.PHONY = Matrix tar clean