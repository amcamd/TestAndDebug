
LDFLAGS=
LD=hipcc
CFLAGS=
CPP=hipcc
OBJ=pack_int8_in_uint32.o
EXE=pack_int8_in_uint32

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
