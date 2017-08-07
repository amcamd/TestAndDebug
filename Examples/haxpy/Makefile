
LDFLAGS=
LD=hipcc
CFLAGS=
CPP=hipcc
OBJ=haxpy.o
EXE=haxpy

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
