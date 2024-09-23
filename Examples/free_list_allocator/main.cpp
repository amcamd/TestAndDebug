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


int main(int argc, char* argv[])
{
    // 
    linked_list<int> my_linked_list;
    linked_list<int>::node my_node;
    my_node.data = 1;
    my_node.next = nullptr;
    my_linked_list.add(nullptr, &my_node);

    free_list_allocator *fl_alloc = new free_list_allocator(4096, free_list_allocator::FIND_FIRST);

    fl_alloc->allocate(512, 64);
    fl_alloc->allocate(64, 64);

    return 0;
}
