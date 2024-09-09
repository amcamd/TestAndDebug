

template <class T>
class linked_list
{
	public:
		struct node
		{
			T data;
			node *next;
		};
		node *head;

	public:
		linked_list();
		void add(node *previous_node, node *new_node);
		void remove(node *previous_node, node *delete_node);
};

template <class T>
linked_list<T>::linked_list(){}

template<class T>
void linked_list<T>::add(node* previous_node, node* new_node){
	if(previous_node == nullptr) // previous_node is first node
	{
		if(head != nullptr) // insert new_node at head of exisisting list
		{
			new_node->next = head;
		}
		else // add new_node is head of empty list
		{
			new_node->next = nullptr;
		}
		head = new_node;
	}
	else
	{
		if(previous_node->next == nullptr) // add new_node at end of list
		{
			previous_node->next = new_node;
			new_node->next = nullptr;
		}
		else  // insert new_node between existing nodes
		{
			new_node->next = previous_node->next;
			previous_node->next = new_node;
		}
	}
}

template<class T>
void linked_list<T>::remove(node* previous_node, node* remove_node)
{
	if(previous_node == nullptr) // remove_node is head
	{
	        head = remove_node->next;
	}
        else
	{
	        previous_node->next = remove_node->next;
	}
}
