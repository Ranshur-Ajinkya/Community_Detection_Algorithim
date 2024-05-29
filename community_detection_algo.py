import numpy as np
from scipy import sparse 
import networkx as nx 
import matplotlib.pyplot as plt

#Function to create an ajacency matrix for testing the code
def create_test_network(n):
    A=np.random.randint(0,2,size=(n,n))
    np.fill_diagonal(A,0) #No self-loops
    return A

def calculate_sign(vectors,values):
#This function is for calculating the value of vector "s" which tell us in which of the two groups does the node belong to by giving it a charge
    if values > 0:
        s=sparse.csc_matrix([[1 if u_1_i > 0 else -1] for u_1_i in vectors])
    return s

def eigen_calculations(B):
#This function is used for calculating the eignevectors and eigenvalues of the vector "B" which is equal to "A_ij - k_ik_j /2m"
    B2= B + B.T
    #print("This is the value of B")
    #print(B)
    values,vectors=np.linalg.eigh(B2)
    #Sort in descending order
    sorted_indices=np.argsort(values)[::-1]
    sorted_values=values[sorted_indices]
    print("Value of sorted matrix")
    print(sorted_values)
    max_eigenvalue=sorted_values[0]
    print("This is the eigenvalue")
    print(max_eigenvalue)
    sorted_vectors=vectors[:,sorted_indices]
    s=calculate_sign(sorted_vectors[:,0],sorted_values[0])
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
    s_array=s.toarray().flatten()
    #Q here is the modularity it tell us whether we can divide this network or not
    Q=np.sum(((np.dot(eigenvectors.T,s_array)) ** 2) * eigenvalues)
    Q=Q/(4 * L) 
    #print("This is the valuof modularity")
    #print(Q)
    #print("This is the eigenvalue")
    #print(eigenvalues)
    return Q,eigenvalues,eigenvectors,s_array

def delta_q(A_positive, A_negative):
    deltaq_add = 0  # Initialize deltaq_add
    if A_positive.size > 0:
        qp, eigenvalues_pos, eigenvectors_pos, s_array_pos = calculate_modularity(A_positive)
        if eigenvalues_pos[0] > 0:
            if qp > 0:
                deltaq_add += qp
                deltaq_add += partition_subnetworks(s_array_pos, A_positive)
            else:
                return 0
        else:
            return 0

    else:
        return 0
    if A_negative.size > 0:
        qn, eigenvalues_neg, eigenvectors_neg, s_array_neg = calculate_modularity(A_negative)
        if eigenvalues_neg[0] > 0:
            if qn > 0:
                deltaq_add += qn
                deltaq_add += partition_subnetworks(s_array_neg, A_negative)
            else:
                return 0
        else:
            return 0
    else:
        return 0
    return deltaq_add

def partition_subnetworks(s_array, A):
    positive_indices = np.where(s_array == 1)[0]
    negative_indices = np.where(s_array == -1)[0]
    A_positive = A[np.ix_(positive_indices, positive_indices)]
    A_negative = A[np.ix_(negative_indices, negative_indices)]
    if A_positive.size <= 0 and A_negative.size <= 0:
        return 0
    return delta_q(A_positive, A_negative)

#def delta_q(A_positive,A_negative):
#    #print("This is the value of A positive")
#    if A_positive.size > 0:
#      qp, eigenvalues_pos, eigenvectors_pos, s_array_pos = calculate_modularity(A_positive)
#      if qp > 0:
#            deltaq_add += qp
#            deltaq_add += partition_subnetworks(s_array_pos, A_positive)
#    if A_negative.size > 0:
#        qn, eigenvalues_neg, eigenvectors_neg, s_array_neg = calculate_modularity(A_negative)
#        if qn > 0:
#            deltaq_add += qn
#            deltaq_add += partition_subnetworks(s_array_neg, A_negative)
#    #print(A_positive)
#    A1=A_positive.copy()
#    A2=A_negative.copy()
#    print("This is the value of the copied A neg matrix")
#    print(A2)
#    qp,eigenvalues_pos,eigenvectors_pos,s_array_pos=calculate_modularity(A1)
#    qn,eigenvalues_neg,eigenvectors_neg,s_array_neg=calculate_modularity(A2)
#
#    if eigenvectors_pos[0]:
#        if qp > 0:
#            deltaq_add+=qp
#            partition_subnetworks(s_array_pos,A1)
#    if eigenvectors_neg[0]:
#        if qn > 0:
#            deltaq_add+=qn
#            partition_subnetworks(s_array_neg,A2)

#    return deltaq_add

#def partition_subnetworks(s_array,A):
#    positive_indices = np.where(s_array == 1)[0]
#    negative_indices = np.where(s_array == -1)[0]
#    A_positive = A[np.ix_(positive_indices, positive_indices)]
#    A_negative = A[np.ix_(negative_indices, negative_indices)]
#   #print("This is the value of the negative variable")
#    #print(A_negative)
#    get_deltaq=delta_q(A_positive,A_negative)
#    #qp,eigenvalues_pos,eigenvectors_pos,s_array_pos=calculate_modularity(A_positive,lamda_=1)
#    #qn,eigenvalues_neg,eigenvectors_neg,s_array_neg=calculate_modularity(A_negative,lamda_=1)
#    return get_deltaq
def main():
    N=10
    A=create_test_network(N)
    Q,eigenvalues,eigenvectors,s_array=calculate_modularity(A)
    if eigenvalues[0] > 0 :
        deltaq_value=partition_subnetworks(s_array,A)
        print("This is the final modularity Value")
        Q=Q+deltaq_value
    else:
        print("This network is not divisible")

if __name__=="__main__":
    main()
