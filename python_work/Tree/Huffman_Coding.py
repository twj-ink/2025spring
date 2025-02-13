import heapq

class Node:
    def __init__(self, freq, char, left = None, right = None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(char_freq):
    heap = [Node(freq, char) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, None) # 合并之后是一个空节点
        merged.left = left
        merged.right = right
        heapq.heappush(heap,merged)

    return heap[0] # root

def weighted_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.freq
    return (weighted_path_length(node.left, depth + 1) +
            weighted_path_length(node.right, depth + 1))

def main():
    char_freq = {'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 8, 'f': 9, 'g': 11, 'h': 12}
    huffman_tree=huffman_encoding(char_freq)
    WPL=weighted_path_length(huffman_tree)
    print('The WPL of the Huffman Tree is:',WPL)

if __name__ == '__main__' :
    main()

# Output:
# The WPL of the Huffman Tree is: 169