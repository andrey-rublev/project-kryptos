import string

def decrypt(ciphertext, key):
    result = ""
    for char in ciphertext:
        if char in string.ascii_lowercase:
            new_index = (ord(char) - ord('a') - key) % 26
            result += chr(new_index + ord('a'))
        else:
            result += char
    return result