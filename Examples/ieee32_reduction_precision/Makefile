LD=g++
CFLAGS=-Wall
CPP=g++
OBJ=main.o
EXE=a.out
LDFLAGS=-Wall

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $@

clean:
	rm -f $(EXE) $(OBJ)
