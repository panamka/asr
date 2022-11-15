import numpy as np
from numpy import log as lg

from utils.text_processing import CharTokenizer


tokenizer = CharTokenizer()
blank_id = tokenizer.text_to_ids('_')[0]

def alligment_forward(matrix, sequence, len_clean_seq):
    t_steps = matrix.shape[-1]
    NEG_INF = -float('inf')

    matrix_path = np.full([len(sequence), t_steps], NEG_INF)

    for t in range(t_steps):
        for unit_id_seq in range(len(sequence)):
            unit = sequence[unit_id_seq]

            #first column
            if t == 0:
                if unit_id_seq < 2:
                    matrix_path[unit_id_seq, t] = lg(matrix[unit], t)
                else:
                    matrix_path[unit_id_seq, t] = NEG_INF
            #upper diagonal zero
            elif unit_id_seq < ((2*len_clean_seq+1) -2*(t_steps - t)):
                matrix_path[unit_id_seq, t] = NEG_INF

            else:
                #first line
                if unit_id_seq == 0:
                    matrix_path[unit_id_seq, t] = matrix_path[unit_id_seq, t-1] + lg(matrix[blank_id], t)
                #second line
                elif unit_id_seq == 1:
                    p_blanc = np.exp(matrix_path[unit_id_seq-1, t-1])
                    p_char = np.exp(matrix_path[unit_id_seq, t-1])
                    p = matrix[unit, t]
                    matrix_path[unit_id_seq, t] = lg((p_blanc + p_char) * p)

                else:
                    prev_unit_proba = np.exp(matrix_path[unit_id_seq - 1, t - 1])
                    prev_step_proba = np.exp(matrix_path[unit_id_seq, t - 1])
                    #rule for blank and same char
                    if unit == sequence[unit_id_seq - 2]:
                        matrix_path[unit_id_seq, t] = lg((prev_unit_proba + prev_step_proba) * matrix[unit, t])
                    else:
                        prev_unit_proba_2 = np.exp(matrix_path[unit_id_seq - 2, t - 1])
                        matrix_path[unit_id_seq, t] = lg((prev_unit_proba + prev_step_proba + prev_unit_proba_2) * matrix[unit, t])
    return matrix_path