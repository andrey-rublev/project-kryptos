import string

def decrypt(ciphertext, key):
    result = ""
    key_index = 0
    key_length = len(key)

    for char in ciphertext:
        if char in string.ascii_lowercase:
            shift = ord(key[key_index % key_length]) - ord('a')
            new_index = (ord(char) - ord('a') - shift) % 26
            result += chr(new_index + ord('a'))
            key_index += 1
        else:
            result += char
    return result