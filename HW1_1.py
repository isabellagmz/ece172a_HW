# Isabella Gomez
# A15305555
# ECE172A

import numpy as np

def hw1_1():

    # -----------------------------------------------------------
    # (i)
    A = np.array([[60, -15, 12, 22, 68], 
        [-97, 28, 91, -49, 7], [9, 57, -91, 91, -88],
        [29, -92, 42, 38, 99], [-58, 91, 90, 95, -62]])
    B = np.array([[1, 1, 0, 1, 1], [1, 0, 1, 1, 0], 
        [1, 0, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 0, 0]])

    print('A: ')
    print(A)
    print('B: ')
    print(B)
    
    # -----------------------------------------------------------
    # (ii)
    locs = {}
    temp_row = 0
    for row in A:
        temp_col = 0
        for cell in row:
            if cell < -70:
                locs[cell] = (str(temp_row), str(temp_col))
            temp_col = temp_col + 1
        temp_row = temp_row + 1

    print('Locations: ')
    print('format: value: (row, col)')
    print(locs)

    # -----------------------------------------------------------
    # (iii)
    C = A*B

    print('C: ')
    print(C)

    # -----------------------------------------------------------
    # (iv)
    inner_product_C = np.inner(C[:,2], C[4,:])

    print('Inner product: ')
    print(inner_product_C)
    
    # -----------------------------------------------------------
    # (v) 

    # this is the location of all the values that equal max_val
    max_val = np.max(C[:,3])
    max_val_loc = np.where(C == max_val)
    print('Locations of max value in C: ')
    print(max_val_loc)


    # this is the location of the max_val on 4th column
    temp_row = 0
    for cell in C[:,3]:
        if cell == max_val:
            max_val_loc = [temp_row, 3]
        temp_row = temp_row + 1

    print('Location of max value in column 4: ')
    print(max_val_loc)
    
    # -----------------------------------------------------------
    # (vi)
    D = C[0,:]*C

    print('D: ')
    print(D)
    
    # -----------------------------------------------------------
    # (vii)
    inner_product_D = np.inner(D[:,2], D[4,:])

    print('Inner product: ')
    print(inner_product_D)

    return
    
if __name__ == '__main__':
    hw1_1()

