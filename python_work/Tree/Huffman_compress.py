from collections import Counter
from heapq import heappop,heappush,heapify

class HNode:
    def __init__(self,val,freq,left=None,right=None):
        self.val=val
        self.freq=freq
        self.left=left
        self.right=right

    def __lt__(self, other):
        return self.freq<other.freq

def build_tree(letter_freq):
    heap=[HNode(letter,freq) for letter,freq in letter_freq.items()]
    heapify(heap)

    while len(heap)>1:
        left=heappop(heap)
        right=heappop(heap)
        merged=HNode(None,left.freq+right.freq)
        merged.left=left
        merged.right=right
        heappush(heap,merged)

    return heap[0]

def generate_huffman_codes(node,prefix='',codebook={}):
    if node is not None:
        if node.val is not None:
            codebook[node.val]=prefix
        generate_huffman_codes(node.left,prefix+'0',codebook)
        generate_huffman_codes(node.right,prefix+'1',codebook)
    return codebook

def compress_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    letter_freq = Counter(text)
    root = build_tree(letter_freq)
    huffman_codes = generate_huffman_codes(root)

    # Compress the text using the Huffman codes
    compressed = ''.join(huffman_codes[char] for char in text)

    # Write the Huffman tree and compressed data into the output file
    with open(output_file, 'wb') as f:
        # Write frequency table (can be used for decompression)
        freq_table = {char: freq for char, freq in letter_freq.items()}
        f.write(bytes(str(freq_table), 'utf-8') + b'\n')

        # Write compressed data (converted to bytes)
        padding = 8 - len(compressed) % 8  # Make the binary string multiple of 8 bits
        compressed = f'{padding:08b}' + compressed  # Add padding information at the start

        # Convert the binary string to bytes
        byte_array = bytearray()
        for i in range(0, len(compressed), 8):
            byte_array.append(int(compressed[i:i + 8], 2))

        f.write(bytes(byte_array))

    return huffman_codes, compressed

def decompress_file(input_file):
    with open(input_file, 'rb') as f:
        # Read the frequency table (first line)
        freq_table = eval(f.readline().decode('utf-8'))

        # Rebuild the Huffman tree
        root = build_tree(freq_table)
        huffman_codes = generate_huffman_codes(root)

        # Read the compressed data (binary stream)
        compressed = f.read()

        # Convert the bytes back into a binary string
        bit_string = ''.join(f'{byte:08b}' for byte in compressed)

        # Remove the padding information from the start of the bit string
        padding = int(bit_string[:8], 2)
        bit_string = bit_string[8:-padding] if padding else bit_string[8:]

        # Decode the bit string using the Huffman codes
        reversed_codes = {v: k for k, v in huffman_codes.items()}
        decoded = []
        buffer = ''
        for bit in bit_string:
            buffer += bit
            if buffer in reversed_codes:
                decoded.append(reversed_codes[buffer])
                buffer = ''

        return ''.join(decoded)

huffman_codes_1,compressed_1 = compress_file('input.txt', 'output.bin')
print(decompress_file('output.bin'))
