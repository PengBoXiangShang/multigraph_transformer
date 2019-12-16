import numpy as np


def produce_adjacent_matrix_random(stroke_length, conn=0.15):
    attention_mask = np.random.choice(a=[0, 1], size=[7, 7], p=[conn, 1-conn])
    attention_mask[stroke_length: , :] = 2
    attention_mask[:, stroke_length : ] = 2
    #####
    attention_mask = np.triu(attention_mask)
    attention_mask += attention_mask.T - np.diag(attention_mask.diagonal())
    
    for i in range(stroke_length):
        attention_mask[i, i] = 0
    return attention_mask




att_msk = produce_adjacent_matrix_random(4, 0.5)
print(att_msk)
print("----------------------")
print((att_msk.T == att_msk).all())