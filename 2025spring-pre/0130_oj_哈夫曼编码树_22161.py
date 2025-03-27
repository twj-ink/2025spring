import heapq

class Node:
    def __init__(self, freq, char, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_tree(char_freq):
    heap=[Node(freq, char) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_tree(root):
    codes={}

    def traversal(node, code):
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traversal(node.left, code + '0')
            traversal(node.right, code + '1')

    traversal(root, '')
    return codes

def encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def decoding(root, string):
    decoded=''
    node = root
    for bit in string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

n = int(input())
char_freq = {}
for _ in range(n):
    char, freq = input().split()
    char_freq[char] = int(freq)

tree = huffman_tree(char_freq)
codes = encode_tree(tree)

strings=[]
while True:
    try:
        l = input()
        strings.append(l)
    except EOFError:
        break

for string in strings:
    if string[0] in ('0', '1'):
        print(decoding(tree, string))
    else:
        print(encoding(codes, string))