
LDFLAGS=
LD=hipcc
CFLAGS=
CPP=hipcc
OBJ=h2axpy_dx4.o
EXE=h2axpy_dx4

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
