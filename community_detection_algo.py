import numpy as np
from scipy import sparse
import traceback
import pandas as pd

def calculate_sign(vectors, values):
    if values > 0:
        s = sparse.csc_matrix(np.sign(vectors).reshape(-1, 1))
    else:
        s = None
    return s

def eigen_calculations(B):
    B2 = B + B.T
    values, vectors = np.linalg.eigh(B2)
    sorted_indices = np.argsort(values)[::-1]
    sorted_values = values[sorted_indices]
    sorted_vectors = vectors[:, sorted_indices]
    s = calculate_sign(sorted_vectors[:, 0], sorted_values[0]) if sorted_values.size > 0 else None
    return sorted_values, sorted_vectors, s

def calculate_modularity(A):
    k_in = np.sum(A, axis=0)
    k_out = np.sum(A, axis=1)
    L = np.sum(A)
    B = A - np.outer(k_in, k_out) / (L)
    eigenvalues, eigenvectors, s = eigen_calculations(B)
    s_array = s.toarray().flatten() if s is not None else np.array([])
    Q = np.sum(((np.dot(eigenvectors.T, s_array)) ** 2) * eigenvalues)
    Q = Q / (4 * L)
    return Q, eigenvalues, eigenvectors, s_array, B, L

def partition_subnetworks(s_array, A):
    positive_indices = np.where(s_array == 1)[0]
    negative_indices = np.where(s_array == -1)[0]
    A_positive = A[np.ix_(positive_indices, positive_indices)]
    A_negative = A[np.ix_(negative_indices, negative_indices)]
    return A_positive, A_negative, positive_indices, negative_indices

def calculate_delta_modularity(A_mod):
    k_in = np.sum(A_mod, axis=0)
    k_out = np.sum(A_mod, axis=1)
    L = np.sum(A_mod)
    B = A_mod - np.outer(k_in, k_out) / (L)
    eigenvalues, eigenvectors, s = eigen_calculations(B)
    s_array = s.toarray().flatten() if s is not None else np.array([])
    Q = np.sum(((np.dot(eigenvectors.T, s_array)) ** 2) * eigenvalues)
    Q = Q / (4 * L)

    return Q

def recursive_modularity(A, labels, depth=0,result_labels_index=1,result_labels=None):
    """
    Recursively calculate modularity and partition the network into communities.

    Args:
        A (numpy.ndarray): The adjacency matrix.
        labels (list): The labels associated with the nodes.
        depth (int, optional): The current depth of recursion. Default is 0.
        result_labels (dict, optional): A dictionary to store the labels of the communities. Default is None.

    Returns:
        tuple: A tuple containing the final modularity value and a dictionary of community labels.
    """
    if result_labels is None:
        result_labels = {}

    try:
        print(f"Iteration Number: {depth}")
        modularity = 0
        if depth == 0:
            init_mod, eigenvalues, eigenvectors, s_array, B_init, numberofedges = calculate_modularity(A)
            modularity += init_mod
        else:
            init_mod, eigenvalues, _, s_array, _, _ = calculate_modularity(A)

        print(f"Current Modularity: {modularity}")

        if np.all(s_array == 1) or np.all(s_array == -1):
            print("Stopping the recurrence as s_array is homogenous")
            result_labels[depth] = labels
            return modularity, result_labels

        if eigenvalues.size > 0 and eigenvalues[0] > 0:
            A_pos, A_neg, pos_indices, neg_indices = partition_subnetworks(s_array, A)
            labels_pos = [labels[i] for i in pos_indices]
            labels_neg = [labels[i] for i in neg_indices]
            print("*****************")
            print("These are the labels Pos",labels_pos)
            print("These are the labels Neg",labels_neg)
            print("*****************")

            if depth == 0:
                recur_q_pos, result_labels = recursive_modularity(A_pos, labels_pos, depth + 1, result_labels)
                recur_q_neg, result_labels = recursive_modularity(A_neg, labels_neg, depth + 1, result_labels)
                modularity += recur_q_pos + recur_q_neg
            else:
                deltaq_pos = calculate_delta_modularity(A_pos)
                deltaq_neg = calculate_delta_modularity(A_neg)
                Q_new = deltaq_pos + deltaq_neg
                delta_mod = init_mod - Q_new

                if delta_mod > 0:
                    if deltaq_pos > 0:
                        modularity += deltaq_pos
                        recur_q_pos, result_labels = recursive_modularity(A_pos, labels_pos, depth + 1, result_labels_index, result_labels)
                        modularity += recur_q_pos


                        result_labels_index = max(result_labels.keys()) + 1
                    else:
                        print("Here are the labels")
                        print(labels_pos)
                        result_labels[result_labels_index] = labels_pos
                        result_labels_index += 1
                        
                    if deltaq_neg > 0:
                        modularity += deltaq_neg
                        recur_q_neg, result_labels = recursive_modularity(A_neg, labels_neg, depth + 1, result_labels_index, result_labels)
                        modularity += recur_q_neg
                    else:
                        print("Here are the labels ")
                        print(labels_neg)
                        result_labels[result_labels_index] = labels_neg
                else:
                    result_labels[depth] = labels_pos
                    result_labels[depth] = labels_neg
                    
                    

        else:
            result_labels[depth] = labels

        return modularity, result_labels

    except Exception as e:
        print(f"An error occurred during recursion at depth {depth}: {e}")
        traceback.print_exc()
        raise

def main():
    try:
        # file_path = r'E:\Summer_Project\Network_Data\connectome_data_anand_pathak\adj_matrix_humanbrain.csv'
        # file_path=r'/workspaces/Community_Detection_Algorithim/adj_matrix_with_labels.csv'
        file_path=r'/workspaces/Community_Detection_Algorithim/adj_matrix_humanbrain.csv'
        adj_matrix_df = pd.read_csv(file_path)

        labels = adj_matrix_df.columns[1:].tolist()
        A = adj_matrix_df.iloc[:, 1:].values
        
        final_modularity, final_label_division = recursive_modularity(A, labels)
        print("This is the final Modularity:", final_modularity)
        
        print("Here are the partitions")
        for subgraph_key, subgraph_labels in final_label_division.items():
            print(f"Subgraph {subgraph_key}: {subgraph_labels}")

    except IOError as e:
        print(f"Error reading file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
