import numpy as np
import math


def givens_rotation_col_major(mat):
    (rows, cols) = mat.shape
    for row_up in range(0, rows):
        col = row_up
        for row_down in range(col+1, rows):
            
            A_mm = mat[row_up, col]
            A_nm = mat[row_down, col]
            r = math.sqrt(A_mm * A_mm + A_nm * A_nm)
            if r < 1e-6:
                continue
            
            c = A_mm / r
            s = A_nm / r
            
            for i in range(col, cols):
                a = mat[row_up, i]
                b = mat[row_down, i]
                mat[row_up, i] = c * a + s * b
                mat[row_down, i] = -s * a + c * b
    return mat


def givens_rotation_row_major(mat):
    (rows, cols) = mat.shape
    
    for row_curr in range(1, rows):
        for col in range(0, min(row_curr, cols)):
            A_mm = mat[col, col]
            A_nm = mat[row_curr, col]
            r = math.sqrt(A_mm * A_mm + A_nm * A_nm)
            if r < 1e-6:
                continue
            
            c = A_mm / r
            s = A_nm / r
            
            row = col
            for i in range(col, cols):
                a = mat[row, i]
                b = mat[row_curr, i]
                mat[row, i] = c * a + s * b
                mat[row_curr, i] = -s * a + c * b
                
    return mat

if __name__=="__main__":
    np.set_printoptions(precision=3)
    mat_input = np.random.rand(4, 5)
    (rows, cols) = mat_input.shape
    mat_givens_rotation = givens_rotation_col_major(mat_input)
    print("matrix after col major givens rotation: \n", mat_givens_rotation)
    mat_givens_rotation = givens_rotation_row_major(mat_input)
    print("matrix after row major givens rotation: \n", mat_givens_rotation)
        

