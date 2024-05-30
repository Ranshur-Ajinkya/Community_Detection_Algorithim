import numpy as np
from scipy import sparse 
import networkx as nx 
import matplotlib.pyplot as plt

def calculate_sign(vectors,values):
#This function is for calculating the value of vector "s" which tell us in which of the two groups does the node belong to by giving it a charge
    if values > 0:
        s=sparse.csc_matrix([[1 if u_1_i > 0 else -1] for u_1_i in vectors])
        #print("Sparse Matrix")
        #print(s)
    return s

def eigen_calculations(B):
#This function is used for calculating the eignevectors and eigenvalues of the vector "B" which is equal to "A_ij - k_ik_j /2m"
    B2= B + B.T
    values,vectors=np.linalg.eigh(B2)
    sorted_indices=np.argsort(values)[::-1]
    sorted_values=values[sorted_indices]
    sorted_vectors=vectors[:,sorted_indices]
    s=calculate_sign(sorted_vectors[:,0],sorted_values[0]) if sorted_values.size > 0 else None
    return sorted_values,sorted_vectors,s

def calculate_modularity(A):
#This function is used for calculating the B vector from the given adjacency matrix
   # print("This is the value of calc mod ")
   # print(A)
    k_in=np.sum(A,axis=0)
    k_out=np.sum(A,axis=1)
    L=np.sum(A)
    #print("Value of in , out , total edges")
    #print(k_in,k_out,L)
    #for Undirected Networks
    B=A - np.outer(k_in,k_out) / (2*L)
    #for directed Networks
    #B = A - np.outer(k_in,k_out) / L
    eigenvalues,eigenvectors,s=eigen_calculations(B)
    s_array=s.toarray().flatten() if s is not None else np.array([])
    #Q here is the modularity it tell us whether we can divide this network or not
    Q=np.sum(((np.dot(eigenvectors.T,s_array)) ** 2) * eigenvalues)
    Q=Q/(4 * L) 
    return Q,eigenvalues,eigenvectors,s_array

#def compute_delta_q(A_matrix):
    #qn, eval_neg, evec_neg, s_array_neg = calculate_modularity(A_negative)
    #qp, eval_pos, evec_pos, s_array_pos = calculate_modularity(A_positive)
    #return qp,qn

def partition_subnetworks(s_array, A):
    print("S Array")
    print(s_array)
    positive_indices = np.where(s_array == 1)[0]
    negative_indices = np.where(s_array == -1)[0]
    A_positive = A[np.ix_(positive_indices, positive_indices)]
    A_negative = A[np.ix_(negative_indices, negative_indices)]
    return A_positive, A_negative


def recursive_modularity(A,Q):

    delta_modularity,eigenvalues,eigenvectors,s_array=calculate_modularity(A)
    Q+=delta_modularity
    if eigenvalues.size> 0 and eigenvalues[0] > 0:
        A_pos,A_neg = partition_subnetworks(s_array,A)
        if A_pos.size==0:
            return Q
        else:
            #deltaq_pos,deltaq_neg=compute_delta_q(A_pos,A_neg)
            deltaq_pos,_,_,_ = calculate_modularity(A_pos)

            if deltaq_pos > 0:
                Q+=deltaq_pos
                new_modularity = recursive_modularity(A_pos,Q)
            else:
                return Q
        if A_neg.size==0:
            return Q
        else:
            deltaq_neg,_,_,_ = calculate_modularity(A_neg)
            if deltaq_neg > 0:
                Q+=deltaq_neg
                new_modularity=recursive_modularity(A_neg,Q)

            return Q

        return Q
    else:
        return Q

def main():
    deltaQ_add=0
    A=np.loadtxt('/media/ajinkyar/coolio/Summer_Project/ucidata-zachary/adj_matrix.csv', delimiter=',',dtype=int)
    Q=0
    final_modularity=recursive_modularity(A,Q)
    print("This is the final Modularity")
    print(final_modularity)
    #Q,eigenvalues,eigenvectors,s_array=calculate_modularity(A)

if __name__=="__main__":
    main()

