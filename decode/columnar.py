import math

def decrypt(ciphertext, key_str):
    key_digits = [int(c) for c in key_str]
    n_cols = len(key_digits)
    n_rows = math.ceil(len(ciphertext) / n_cols)

    # Pad ciphertext if necessary
    padded_length = n_rows * n_cols
    text = ciphertext.ljust(padded_length, 'x')

    # Build empty grid
    grid = [ [''] * n_cols for _ in range(n_rows) ]

    sorted_key = sorted(key_digits)
    start = 0
    for digit in sorted_key:
        col_index = key_digits.index(digit)
        for row in range(n_rows):
            grid[row][col_index] = text[start]
            start += 1

    # Read row-wise
    plaintext = "".join("".join(row) for row in grid)
    # remove padding x's if needed
    plaintext = plaintext.rstrip('x')
    return plaintext