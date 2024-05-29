mport numpy as np
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
        #print("Sparse Matrix")
        #print(s)
    return s

def eigen_calculations(B):
#This function is used for calculating the eignevectors and eigenvalues of the vector "B" which is equal to "A_ij - k_ik_j /2m"
    B2= B + B.T
    #print("This is the value of B")
    #print(B)
    values,vectors=np.linalg.eigh(B2)
    #print("These are the vectors")
    #print(vectors)
    #print("These are the values")
    #print(values)
    #Sort in descending order
    sorted_indices=np.argsort(values)[::-1]
    sorted_values=values[sorted_indices]
    #print("Value of sorted matrix")
    #print(sorted_values)
    max_index=np.argmax(values)
    max_eigenvalue=values[max_index]
    max_eigenvector=vectors[:,max_index]
    #print("This is the eigenvector")
    #print(max_eigenvector)
    sorted_vectors=vectors[:,sorted_indices]
    print("Sorted Vectors")
    print(sorted_vectors)
    #print("Sorted Values")
    #print(sorted_values[0])
    s=calculate_sign(sorted_vectors[0],sorted_values[0])
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
    qn, eval_neg, evec_neg, s_array_neg = calculate_modularity(A_negative)
    qp, eval_pos, evec_pos, s_array_pos = calculate_modularity(A_positive)
    return qp, qn

def partition_subnetworks(s_array, A):
    print("S Array")
    print(s_array)
    positive_indices = np.where(s_array == 1)[0]
    negative_indices = np.where(s_array == -1)[0]
    A_positive = A[np.ix_(positive_indices, positive_indices)]
    A_negative = A[np.ix_(negative_indices, negative_indices)]
    #print("Partition Subnetwork")
    #print(A_positive)
    return A_positive, A_negative

def main():
    N=10
    A=create_test_network(N)
    Q,eigenvalues,eigenvectors,s_array=calculate_modularity(A)
    if eigenvalues[0] > 0 :
        A_pos,A_neg=partition_subnetworks(s_array,A)
       # print("A positive")
       # print(A_pos)
       # print("A Negative")
       # print(A_neg)
        deltaq_pos,deltaq_neg=delta_q(A_pos,A_neg)
        print("Delta Pos")
        print(deltaq_pos)
        print("Delta Neg")
        print(deltaq_neg)
        #Q=Q+deltaq_pos+delta_neg
        #print(Q)
    else:
        print("This network is not divisible")

if __name__=="__main__":
    main()

