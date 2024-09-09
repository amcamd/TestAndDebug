#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <random>
#include <limits>
#include <cstring>
#include <stdlib.h>
#include <chrono>
#include "linked_list.hpp"
#include "free_list_allocator.hpp"

void funct(size_t arg1)
{
    std::cout << "arg1 = " << arg1 << std::endl;
}

int main(int argc, char* argv[])
{
    size_t int1 = 100;
    float extra_mem = 1.5;

    funct(int1 * extra_mem);

    
    std::cout << "hello, hope this works" << std::endl;

    linked_list<int> my_linked_list;
    linked_list<int>::node my_node;
    my_node.data = 1;
    my_node.next = nullptr;
    my_linked_list.add(nullptr, &my_node);

    free_list_allocator(1024, free_list_allocator::FIND_FIRST);

    return 0;
}
