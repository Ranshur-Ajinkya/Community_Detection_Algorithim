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
    lambda_=0.4
    if L == 0:
        return 0, None, None, np.array([]), None, L

    B = A - lambda_ * (np.outer(k_in, k_out) / L)
    eigenvalues, eigenvectors, s = eigen_calculations(B)
    s_array = s.toarray().flatten() if s is not None else np.array([])
    if s_array.size == 0:
        return 0, eigenvalues, eigenvectors, s_array, B, L

    # Q = np.sum(((np.dot(eigenvectors.T, s_array)) ** 2) * eigenvalues)
    Q = s_array.T @ B @ s_array
    Q = Q / (1 * L)
    return Q, eigenvalues, eigenvectors, s_array, B, L

def partition_subnetworks(s_array, A):
    positive_indices = np.where(s_array == 1)[0]
    negative_indices = np.where(s_array == -1)[0]
    A_positive = A[np.ix_(positive_indices, positive_indices)] if positive_indices.size > 0 else np.array([[]])
    A_negative = A[np.ix_(negative_indices, negative_indices)] if negative_indices.size > 0 else np.array([[]])
    return A_positive, A_negative, positive_indices, negative_indices

def calculate_delta_modularity(A_mod):
    if A_mod.size == 0:
        return 0
    
    k_in = np.sum(A_mod, axis=0)
    k_out = np.sum(A_mod, axis=1)
    L = np.sum(A_mod)
    if L == 0:
        return 0
    lambda_=0.4
    B = A_mod - lambda_ * (np.outer(k_in, k_out) / L)
    eigenvalues, eigenvectors, s = eigen_calculations(B)
    s_array = s.toarray().flatten() if s is not None else np.array([])
    if s_array.size == 0:
        return 0

    # Q = np.sum(((np.dot(eigenvectors.T, s_array)) ** 2) * eigenvalues)
    Q=s_array.T @ B @ s_array
    Q = Q / (1 * L)
    return Q

def recursive_modularity(A, labels, depth=0, result_community=None):
    if result_community is None:
        result_community = {1: labels}

    modularity = 0
    improvement = True

    while improvement:
        improvement = False
        communities_to_split = list(result_community.items())

        for community, community_labels in communities_to_split:
            community_indices = [labels.index(label) for label in community_labels]
            sub_community = A[np.ix_(community_indices, community_indices)]
            Q, eigenvalues, eigenvectors, s_array, B, L = calculate_modularity(sub_community)
            if np.all(s_array == 1) or np.all(s_array == -1) or s_array.size == 0:
                continue

            A_pos, A_neg, pos_indices, neg_indices = partition_subnetworks(s_array, sub_community)
            deltaq_pos = calculate_delta_modularity(A_pos)
            deltaq_neg = calculate_delta_modularity(A_neg)
            Q_new = deltaq_pos + deltaq_neg

            if Q_new > Q:
                modularity += Q_new
                pos_labels = [community_labels[i] for i in pos_indices]
                neg_labels = [community_labels[i] for i in neg_indices]
                result_community[community] = pos_labels
                result_community[max(result_community.keys()) + 1] = neg_labels
                improvement = True
            else:
                modularity += Q

    return modularity, result_community

def main():
    try:
        file_path = r'/workspaces/Community_Detection_Algorithim/adj_matrix_humanbrain.csv'
        adj_matrix_df = pd.read_csv(file_path)
        labels = adj_matrix_df.columns[1:].tolist()
        A = adj_matrix_df.iloc[:, 1:].values
        final_modularity, final_labels = recursive_modularity(A, labels)
        # print(len(final_labels))
        # print(final_labels)
        for key,value in final_labels.items():
            print(key,value)
            print("*******")
    except IOError as e:
        print(f"Error reading file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
