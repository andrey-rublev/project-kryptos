def decrypt(ciphertext, key):
    n = len(ciphertext)
    if n == 0:
        return ""
    
    result = [""] * n
    index = 0
    for char in ciphertext:
        result[index] = char
        index = (index + key) % n
    return "".join(result)