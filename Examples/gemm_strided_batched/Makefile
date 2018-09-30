EXE=rocblas-bench
OBJ=bench_to_test.o
CPP=g++
LD=g++

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
