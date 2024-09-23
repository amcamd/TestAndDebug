#include <cassert>    // assert
#include <algorithm>  // std::max

size_t calculate_padding(const size_t base_address, const size_t alignment)
{
    const size_t mult = (base_address / alignment) + 1;
    const size_t aligned_address = mult * alignment;
    const size_t padding = aligned_address - base_address;
    return padding;
}

size_t calc_padding_with_header(const size_t base_address, const size_t alignment, const size_t header_size)
{
    size_t padding = calculate_padding(base_address, alignment);
    if(padding < header_size)
    {
        size_t required_space = header_size - padding;
        if(required_space % alignment == 0)
        {
            padding += alignment * (required_space / alignment);
        }
        else
        {
            padding += alignment * ( 1 + (required_space / alignment));
        }
    }
    return padding;
}

class free_list_allocator
{
    public:
        enum placement_policy{FIND_FIRST, FIND_BEST };

        free_list_allocator(const size_t total_size, const placement_policy p_policy);

        ~free_list_allocator();

        void* allocate(const size_t size, const size_t alignment=0);
        void free(void* ptr);

        void reset();
        void init();
    private:
        placement_policy m_p_policy;
        size_t m_total_size;
        size_t m_used;
        size_t m_peak;
        void* m_start_ptr = nullptr;

        struct free_header{size_t blocksize;};
        struct allocation_header{ size_t blocksize; char padding;};

        linked_list<free_header> m_freelist;

        typedef linked_list<free_header>::node fh_node;

        void       find(const size_t size, const size_t alignment, size_t& padding, 
                               fh_node *& previous_node, fh_node *& found_node);
        void find_first(const size_t size, const size_t alignment, size_t& padding, 
                               fh_node *& previous_node, fh_node *& found_node);
        void  find_best(const size_t size, const size_t alignment, size_t& padding, 
                               fh_node *& previous_node, fh_node *& found_node);
        void coalesce(fh_node *previous_node, fh_node *free_node);

};

free_list_allocator::free_list_allocator(const size_t total_size, const placement_policy p_policy)
    : m_total_size{total_size}, m_used{0}, m_peak{0}
{
    m_p_policy = p_policy;

    if(m_start_ptr != nullptr)
    {
        free(m_start_ptr);
        m_start_ptr = nullptr;
    }
    m_start_ptr = malloc(m_total_size);
    this->reset();
}

free_list_allocator::~free_list_allocator()
{
    free(m_start_ptr);
    m_start_ptr = nullptr;
}

void free_list_allocator::reset()
{
    m_used = 0;
    m_peak = 0;

    fh_node* first_node = (fh_node *) m_start_ptr;
    first_node->data.blocksize = m_total_size;
    first_node->next = nullptr;
    m_freelist.head = nullptr;
    m_freelist.add(nullptr, first_node);
}

void free_list_allocator::init()
{
    if(m_start_ptr != nullptr)
    {
        std::cout << "WARNING: init is freeing memory" << std::endl;
        free(m_start_ptr);
        m_start_ptr = nullptr;
    }
    m_start_ptr = malloc(m_total_size);
    this->reset();
}

void free_list_allocator::find(const size_t size, const size_t alignment, size_t& padding, 
                               fh_node *& previous_node, fh_node *& found_node)
{
    if(m_p_policy == FIND_FIRST)
    {
        find_first(size, alignment, padding, previous_node, found_node);
    }
    else if(m_p_policy == FIND_BEST)
    {
        find_best(size, alignment, padding, previous_node, found_node);
    }
}

void free_list_allocator::find_first(const size_t size, const size_t alignment, size_t& padding,
                                     fh_node *& previous_node, fh_node *& found_node)
{
    fh_node *it = m_freelist.head;
    fh_node *it_prev = nullptr;

    while(it != nullptr)
    {
        padding = calc_padding_with_header((size_t) it, alignment, sizeof(free_list_allocator::allocation_header));
        const size_t required_space = size + padding;
        if(it->data.blocksize >= required_space) { break; }

        it_prev = it;
        it=it->next;
    }
    previous_node = it_prev;
    found_node = it;
}                                     

void free_list_allocator::find_best(const size_t size, const size_t alignment, size_t& padding,
                                    fh_node *& previous_node, fh_node *& found_node)
{
    std::cout << "not implemented" << std::endl;
}                                     

void* free_list_allocator::allocate(const size_t size, const size_t alignment)
{
    const size_t allocation_header_size = sizeof(free_list_allocator::allocation_header);
    const size_t free_header_size = sizeof(free_list_allocator::free_header);
    assert(size >= sizeof(fh_node));
    assert(alignment >= 8);

    size_t padding;
    fh_node *affected_node;
    fh_node *previous_node;
    this->find(size, alignment, padding, previous_node, affected_node);

    assert(affected_node != nullptr && "Not enough memory");

    const size_t alignment_padding = padding - allocation_header_size;
    const size_t required_size = size + padding;

    const size_t excess_size = affected_node->data.blocksize - required_size;

    if(excess_size > 0)
    {
        fh_node *new_node = (fh_node *) ((size_t)affected_node + required_size);
        new_node->data.blocksize =  excess_size;
        m_freelist.add(affected_node, new_node);
    }
    m_freelist.remove(previous_node, affected_node);

    const size_t header_address = (size_t)affected_node + alignment_padding;
    const size_t data_address = header_address + allocation_header_size;
    ((free_list_allocator::allocation_header *) header_address)->blocksize = required_size;
    ((free_list_allocator::allocation_header *) header_address)->padding = alignment_padding;

    m_used += required_size;
    m_peak = std::max(m_peak, m_used);

    std::cout << "Aloc" << "\t@header " << (void*) header_address << "\t@data " << (void*) data_address 
              << "\tblocksize " << ((free_list_allocator::allocation_header *)header_address)->blocksize
              << "\tAP " << alignment_padding << "\tP " << padding << "\tused " << m_used << "\texcess " 
              << excess_size << std::endl;

    return (void*) data_address;
}

void free_list_allocator::coalesce(fh_node *previous_node, fh_node *free_node)
{
    if(free_node->next != nullptr && (size_t(free_node) + free_node->data.blocksize == size_t(free_node->next)))
    {
        free_node->data.blocksize += free_node->next->data.blocksize;
        m_freelist.remove(free_node, free_node->next);
        std::cout << "\tmerging " << (void*)free_node << " & " << (void*)(free_node->next) << "\tblocksize " 
                  << free_node->data.blocksize << std::endl;
    }
    if(previous_node != nullptr && (size_t(previous_node) + previous_node->data.blocksize == size_t(free_node)))
    {
        previous_node->data.blocksize += free_node->data.blocksize;
        m_freelist.remove(previous_node, free_node);
        std::cout << "\tmerging " << (void*)previous_node << " & " << (void*)free_node << "\tblocksize " 
                  << previous_node->data.blocksize << std::endl;
    }
}

void free_list_allocator::free(void* ptr)
{
    const size_t current_address = size_t(ptr);
    const size_t header_address = current_address - sizeof(free_list_allocator::allocation_header);
    const free_list_allocator::allocation_header *allocation_header{ (free_list_allocator::allocation_header*)header_address};

    fh_node *free_node = (fh_node*)header_address;
    free_node->data.blocksize = allocation_header->blocksize + allocation_header->padding;
    free_node->next = nullptr;

    fh_node *it = m_freelist.head;
    fh_node *it_prev = nullptr;
    while(it != nullptr)
    {
        if(ptr < it)
        {
            m_freelist.add(it_prev, free_node);
            break;
        }
        it_prev = it;
        it = it->next;
    }
    m_used -= free_node->data.blocksize;

    coalesce(it_prev, free_node);

    std::cout << "free" << "\t@ptr " << ptr << "\tH@ " << (void*)free_node << "\tS " << free_node->data.blocksize 
              << "\tM " << m_used << std::endl;

}
