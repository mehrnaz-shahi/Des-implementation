# Initial Permutation Table
initial_perm = [58, 50, 42, 34, 26, 18, 10, 1,
                60, 52, 44, 36, 28, 20, 12, 4,
                62, 54, 46, 38, 30, 22, 14, 6,
                64, 56, 48, 40, 32, 24, 16, 8,
                57, 49, 41, 33, 25, 17, 9, 2,
                59, 51, 43, 35, 27, 19, 11, 3,
                61, 53, 45, 37, 29, 21, 13, 5,
                63, 55, 47, 39, 31, 23, 15, 7]

# Expansion P-box Table
exp_p = [32, 1, 2, 3, 4, 5, 4, 5,
         6, 7, 8, 9, 8, 9, 10, 11,
         12, 13, 12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21, 20, 21,
         22, 23, 24, 25, 24, 25, 26, 27,
         28, 29, 28, 29, 30, 31, 32, 1]

# S-box Table
s_box = [
    # s-box 1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # s-box 2
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    # s-box 3
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    # s-box 4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    # s-box 5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # s-box 6
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    # s-box 7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # s-box 8
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

# Final Permutation Table
final_perm = [8, 40, 48, 16, 56, 24, 64, 32,
              39, 7, 47, 15, 55, 23, 63, 31,
              38, 6, 46, 14, 54, 22, 62, 30,
              37, 5, 45, 13, 53, 21, 61, 29,
              36, 4, 44, 12, 52, 20, 60, 28,
              35, 3, 43, 11, 51, 19, 59, 27,
              34, 2, 42, 10, 50, 18, 58, 26,
              33, 1, 41, 9, 49, 17, 57, 25]

# parity bit drop table
keyp = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

# Key Compression Table
key_comp = [14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32]


class DES:

    def __init__(self):
        # Initial Permutation Table
        self.initial_perm = initial_perm

        # Expansion P-box Table
        self.exp_p = exp_p

        # S-box Table
        # every element - 1
        self.s_box = s_box

        # Final Permutation Table
        self.final_perm = final_perm

        # key parity drop
        self.key_parity = keyp

        # key compression
        self.key_comp = key_comp

        self.sp_box = []

        self.data = []

    # convert string to bits
    def str_to_bits(self, str):
        return ''.join(format(ord(c), '08b') for c in str)

    def binary_to_ascii(self, binary_string):
        # Split the binary string into groups of 8 bits
        binary_groups = [binary_string[i:i + 8] for i in range(0, len(binary_string), 8)]

        # Convert each binary group to decimal and then ASCII character
        ascii_chars = [chr(int(group, 2)) for group in binary_groups]

        # Join the ASCII characters to form the final string
        ascii_string = ''.join(ascii_chars)

        return ascii_string

    def bits_to_str(self, bits):
        arr = []
        for i in range(0, len(bits) + 1, 8):
            temp = bits[i: i + 8]
            arr.append(self.binary_to_ascii(temp))

        return "".join([char for char in arr if char.isprintable()])

    # add padding to str less than 64 bit
    def pad_text(self, text):
        padding_length = 8 - (len(text) % 8)
        padding = chr(padding_length) * padding_length
        return text + padding

    def apply_permutation(self, original_array, permutation):
        return ''.join([original_array[permutation[i] - 1] for i in range(len(permutation))])

    def xor_blocks(self, block1, block2, padding=64):
        # Convert each block to a int
        bin_block1 = int(block1, 2)
        bin_block2 = int(block2, 2)

        # XOR
        xor_result = bin_block1 ^ bin_block2

        # Convert the resulting integer to a binary string and pad with leading zeros to ensure it has a length of 64
        # bits
        bin_result = bin(xor_result)[2:].zfill(padding)

        # Return the binary string
        return bin_result

    def split_block(self, block):
        return block[:len(block) // 2], block[len(block) // 2:]

    def combine_blocks(self, left, right):
        return left + right

    def apply_sbox(self, block, s_box_num):
        row = int(str(block[0]) + str(block[5]), 2)
        col = int("".join([str(x) for x in block[1:][:-1]]), 2)
        val = self.s_box[s_box_num][row][col]

        return ''.join([x for x in list('{0:04b}'.format(val))])

    def left_shift(self, data):
        return data[1:] + data[:1]

    def key_generation(self, main_key):
        key = self.apply_permutation(main_key, self.key_parity)
        key_left, key_right = self.split_block(key)
        key_left = self.left_shift(key_left)
        key_right = self.left_shift(key_right)
        key = self.combine_blocks(key_left, key_right)
        key = self.apply_permutation(key, self.key_comp)

        return key

    def hex_to_bit(selfe, str):
        return bin(int(str, 16))[2:].zfill(len(str) * 4)

    def find_all_indices(self, str, char):
        return [i for i in range(len(str)) if str[i] == char]

    def find_p_box_output_input(self, plain, cipher, main_key):

        # print(f'plain is {plain}  && ci[her : {cipher}')
        binary_plain = self.str_to_bits(self.pad_text(plain))

        binary_cipher = self.hex_to_bit(cipher)

        binary_key = self.hex_to_bit(main_key)
        key = self.key_generation(binary_key)

        num = len(binary_plain) // 64

        for i in range(num):

            result = ''

            plain_text = binary_plain[(i * 64): (i + 1) * 64]
            cipher_text = binary_cipher[(i * 64): (i + 1) * 64]
            after_initial = self.apply_permutation(plain_text, initial_perm)
            cipher_after_initial = self.apply_permutation(cipher_text, initial_perm)
            left, right = self.split_block(after_initial)
            cleft, cright = self.split_block(cipher_after_initial)
            # Output of function
            cleft_xor_left = self.xor_blocks(cleft, left, 32)
            # Function
            # expansion p_box
            exp = self.apply_permutation(right, exp_p)
            xor_to_key = self.xor_blocks(exp, key, 48)
            for i in range(0, len(xor_to_key), 6):
                s_num = i // 6
                sbox_res = self.apply_sbox(xor_to_key[i: i + 6], s_num)
                result += sbox_res

            self.data.append([result, cleft_xor_left])

    def encrypt_decrypt(self, text, main_key, type):

        if type == 'en':
            binary = self.str_to_bits(self.pad_text(text))
        if type == 'de':
            binary = self.hex_to_bit(text)

        binary_key = self.hex_to_bit(main_key)
        key = self.key_generation(binary_key)

        # Number of blocks (each block has 64 bit)
        num = len(binary) // 64

        final_result = ''
        for i in range(num):

            result = ''

            block = binary[(i * 64): (i + 1) * 64]
            after_initial = self.apply_permutation(block, initial_perm)
            left, right = self.split_block(after_initial)
            # Function
            # expansion p_box
            exp = self.apply_permutation(right, exp_p)
            xor_to_key = self.xor_blocks(exp, key, 48)

            for i in range(0, len(xor_to_key), 6):
                s_num = i // 6
                sbox_res = self.apply_sbox(xor_to_key[i: i + 6], s_num)
                result += sbox_res

            # straight p_box
            out = self.apply_permutation(result, self.sp_box)
            left_xor_out = self.xor_blocks(left, out, 32)

            output = self.combine_blocks(left_xor_out, right)

            output = self.apply_permutation(output, final_perm)
            final_result += output

        return self.bits_to_str(final_result)

    # Find all permutations based input and output
    def find_permute(self, input, output):
        res = []

        for i in range(len(output)):
            matches = []

            index = -1

            while True:
                try:
                    # Find next occurrence of character starting after previous index
                    index = input.index(output[i], index + 1)
                    matches.append(index)
                except ValueError:
                    # No more occurrences found
                    break

            res.append(matches)

        return res

    def find_common_elements(self, list1, list2):
        common_element_resulte = []
        for i in range(len(list1)):
            common_elements = []
            for element in list1[i]:
                if element in list2[i]:
                    common_elements.append(element)

            common_element_resulte.append(common_elements)

        return common_element_resulte

    def final_straight_pbox(self, initial_permute):

        first_exampl = initial_permute[0]

        for item in initial_permute:
            first_exampl = des.find_common_elements(first_exampl, item)

        straight_pbox = []

        for sublist in first_exampl:
            n = 0
            for i in range(len(sublist)):
                if sublist[i] not in straight_pbox:
                    straight_pbox.append(sublist[i])
                    n = 0
                    break

        return [x + 1 for x in straight_pbox]


if __name__ == '__main__':
    des = DES()

    key = '4355262724562343'

    ciphertext = input()

    plain_and_cipher = {
        'kootahe': '6E2F7B25307C3144',
        'Zendegi': 'CF646E7170632D45',
        'Edame': 'D070257820560746',
        'Dare': '5574223505051150',
        'JolotYe': 'DB2E393F61586144',
        'Daame': 'D175257820560746',
        'DaemKe': 'D135603D1A705746',
        'Mioftan': 'D83C6F7321752A54',
        'Toosh': '413A2B666D024747',
        'HattaMo': '5974216034186B44',
        'khayeSa': 'EA29302D74463545',
        '05753jj': 'B1203330722B7A04',
        '==j95697': '38693B6824232D231D1C0D0C4959590D',
    }

    # Complete the inputs & outputs of P-box
    for plain, cipher in plain_and_cipher.items():
        des.find_p_box_output_input(plain, cipher, key)

    permut_result = []

    for item in des.data:
        inpu, value = item[0], item[1]
        result = des.find_permute(inpu, value)
        permut_result.append(result)

    straight_pbox = des.final_straight_pbox(permut_result)

    des.sp_box = straight_pbox

    print(des.encrypt_decrypt(ciphertext, key, 'de'))
