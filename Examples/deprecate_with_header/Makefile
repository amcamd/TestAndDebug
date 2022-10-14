all: v1.exe v2.exe

v1.exe: main_v1.o mylib.a
	g++ -o v1.exe main_v1.o mylib.a

v2.exe: main_v2.o mylib.a
	g++ -o v2.exe main_v2.o mylib.a

main_v1.o: main_v1.cpp my_function_v1.h
	g++ -c main_v1.cpp 

mylib.a: my_functions.o
	ar rcs mylib.a my_functions.o

clean:
	rm -f v1.exe v2.exe main_v1.o main_v2.o function.o mylib.a
