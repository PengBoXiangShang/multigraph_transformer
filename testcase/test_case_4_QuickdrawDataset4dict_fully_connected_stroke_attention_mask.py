import numpy as np
import ipdb


def detect_ending_points(flag_bits, stroke_len):

    assert flag_bits.shape == (10, 1)
    ending_points_index_array = []
    # 
    ending_points_index_array.append(-1)
    
    for idx in range(stroke_len):
        # 
        if (flag_bits[idx] == 101):
            ending_points_index_array.append(idx)

    # 
    if stroke_len == 10:

        assert flag_bits[9] == 100 or flag_bits[9] == 101

        if flag_bits[9] == 100:
            ending_points_index_array.append(9)

    return ending_points_index_array

def produce_adjacent_matrix(ending_points_index_array):
    adja_matr = np.zeros([10, 10], int)
    
    adja_matr[ : ][ : ] = -1e10
    for idx in range(1, len(ending_points_index_array)):
        start_index = ending_points_index_array[idx - 1] + 1
        end_index = ending_points_index_array[idx]
        #
        if end_index == 9:
            adja_matr[start_index : , start_index : ] = 0
        else:
            adja_matr[start_index : end_index + 1, start_index: end_index + 1] = 0

    return adja_matr



####################




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


#coordi_array = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 0, 1], [9, 9, 1, 0], [10, 10, 0, 1]], dtype=np.float32)

coordi_array = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [3, 3, 0, 1], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0], [7, 7, 1, 0], [8, 8, 0, 1], [9, 9, 1, 0], [10, 10, 1, 0]], dtype=np.float32)



stroke_len = stroke_length_detection(coordi_array)
flag_bits = flag_bit_transfer(coordi_array[:, 2 : ])



ending_points_index_array = detect_ending_points(flag_bits, stroke_len)
attention_mask = produce_adjacent_matrix(ending_points_index_array)



ipdb.set_trace()