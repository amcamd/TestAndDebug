

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
        void reset();
        void init();
    private:
        size_t m_total_size;
        size_t m_used;
        size_t m_peak;
        placement_policy m_p_policy;
        struct free_header{size_t blocksize;};
        struct allocation_header{ size_t block_size; char padding;};

        void* m_start_ptr = nullptr;
        linked_list<free_header> m_freelist;
        void       find(const size_t size, const size_t alignment, size_t& padding, 
                               linked_list<free_header>::node *& previous_node,
                               linked_list<free_header>::node *& found_node);
        void find_first(const size_t size, const size_t alignment, size_t& padding, 
                               linked_list<free_header>::node *& previous_node,
                               linked_list<free_header>::node *& found_node);
        void  find_best(const size_t size, const size_t alignment, size_t& padding, 
                               linked_list<free_header>::node *& previous_node,
                               linked_list<free_header>::node *& found_node);
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

    linked_list<free_header>::node* first_node = (linked_list<free_header>::node *) m_start_ptr;
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
                               linked_list<free_header>::node *& previous_node,
                               linked_list<free_header>::node *& found_node)
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
                               linked_list<free_header>::node *& previous_node,
                               linked_list<free_header>::node *& found_node)
{
    std::cout << "not implemented" << std::endl;
}                                     

void free_list_allocator::find_best(const size_t size, const size_t alignment, size_t& padding,
                               linked_list<free_header>::node *& previous_node,
                               linked_list<free_header>::node *& found_node)
{
    std::cout << "not implemented" << std::endl;
}                                     
