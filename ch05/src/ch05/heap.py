class Heap:
    """An implementation of a heap data structure
    
    A heap is a binary tree, which is represented by an array A with two
    attributes:
     - A.length: Number of elements in the array
     - A.heap_size: Number of elements in the array
    """
    def __init__(self, elements=None):
        self.A = list(elements) if elements is not None else []
        self.heap_size = len(self.A) # Valid element count
        self.length = len(self.A)    # Heap size

    @staticmethod
    def parent(i):
        return (i - 1) >> 1

    @staticmethod
    def left(i):
        return (i << 1) + 1

    @staticmethod
    def right(i):
        return (i << 1) + 2

    def __str__(self) -> None:
        return str(self.A)

    def max_heapify(self, i: int) -> None:
        """Maintains the max-heap property rooted at index i in O(lg n) time."""
        l = self.left(i)
        r = self.right(i)
        largest = i

        # Compare element with left child
        if l < self.heap_size and self.A[l] > self.A[largest]:
            largest = l

        # Compare element with right child
        if r < self.heap_size and self.A[r] > self.A[largest]:
            largest = r

        # If a child has a larger value, swap and recursively heapify
        if largest != i:
            self.A[i], self.A[largest] = self.A[largest], self.A[i]
            self.max_heapify(largest)

    def build_max_heap(self) -> None:
        """Builds a max-heap in linear time from the unordered array."""
        self.heap_size = len(self.A)
        self.length = len(self.A)
        # Heapify down from the last internal node down to the root
        for i in range((len(self.A) // 2) - 1, -1, -1):
            self.max_heapify(i)

if __name__ == '__main__':
    # heap = Heap([0,1,2,3,4,5,6])
    heap = Heap([6,5,4,3,2,1,0])
    heap.build_max_heap()

    print(heap)