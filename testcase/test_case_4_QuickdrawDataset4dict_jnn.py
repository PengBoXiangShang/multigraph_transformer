import numpy as np
import ipdb





def produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (10, 1)
    adja_matr = np.zeros([10, 10], int)
    adja_matr[ : ][ : ] = -1
    
    adja_matr[0][0] = 0
    adja_matr[0][stroke_len - 1] = 0
    adja_matr[stroke_len - 1][stroke_len - 1] = 0
    adja_matr[stroke_len - 1][0] = 0

    assert flag_bits[0] == 100 or flag_bits[0] == 101

    if (flag_bits[0] == 101) and stroke_len >= 2:
        adja_matr[0][1] = 0


    for idx in range(1, stroke_len):

        assert flag_bits[idx] == 100 or flag_bits[idx] == 101

        adja_matr[idx][idx] = 0

        if (flag_bits[idx - 1] == 101):
            adja_matr[idx][idx - 1] = 0

        if (idx == stroke_len - 1):
            break

        # 
        if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 101):
            adja_matr[idx][idx + 1] = 0


    return adja_matr


def stroke_length_detection(coordinate_array):

    for i in range(len(coordinate_array)-1,-1,-1):
        #ipdb.set_trace()

        if ((coordinate_array[i] == np.array([0, 0, 0, 0])).all()):
 
            return i
            
    return 10


def flag_bit_transfer(input_array):
    out_array = np.zeros([10, 1], int)
    assert input_array.shape == (10, 2)
    for idx, bits in enumerate(input_array):
        if ((bits == [1, 0]).all()):
            out_array[idx] = 100
        elif ((bits == [0, 1]).all()):
            out_array[idx] = 101
        else:
            out_array[idx] = 102
    
    return out_array


#coordi_array = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 0, 1], [0, 0, 0, 0], [-1, -1, -1, -1]], dtype=np.float32)

#coordi_array = np.array([[1, 1, 0, 1], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 0, 1], [0, 0, 0, 0], [-1, -1, -1, -1]], dtype=np.float32)


# coordi_array = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 0, 1], [9, 9, 1, 0], [10, 10, 0, 1]], dtype=np.float32)

#coordi_array = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 1, 0], [9, 9, 0, 1], [10, 10, 0, 1]], dtype=np.float32)


coordi_array = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 0, 1], [9, 9, 1, 0], [10, 10, 1, 0]], dtype=np.float32)



stroke_len = stroke_length_detection(coordi_array)
flag_bits = flag_bit_transfer(coordi_array[:, 2 : ])

adja_matr = produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_len)

x = adja_matr

assert (x.T == x).all() == True

ipdb.set_trace()