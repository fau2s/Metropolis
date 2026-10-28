import numpy as np
# Generate VAR(1) data with given parameters
# s(t+1)=A*s(t)+sigma*Z(t)
# A is generated randomly but needs to be hermitian with small eigvalues
# @author: Norbert Fogarasi


def generateVAR1(n,t,sigma):
    # eg. n=15; t=1000; sigma=0.1;
    #############################
    ## Set up sample A matrix
    #############################
    # generate U*V' orthonormal basis
    done = False
    # Try until we find A and c such that the generated sequence is all
    # positive
    kk = 1
    while not done:
        kk += 1
        print(kk,'\n')
        M = np.random.randn(n,n)
        [U, S, V] = np.linalg.svd(np.random.randn(n,n))
        # eigenvalue diagonal matrix
        D = np.diagflat(np.random.rand(n,1))
        #orthogonal matrix
        O = np.matmul(U,np.transpose(V))
        A = np.matmul(D,np.linalg.inv(O))
        A = np.matmul(O,A)
        #A = np.matmul(U, np.matmul(np.transpose(V), np.matmul(D, np.linalg.inv(np.matmul(U, np.transpose(V))))))
        s = np.ones((n,t))

        # generate s matrix from A plus noise11
        for c in np.arange(0, 2, 0.1):
            for i in range(t - 1):
                # add constant at each step to stay positive
                s[:, i+1] = A.dot(s[:, i]) + np.random.normal(0, sigma, n) + c * np.ones((1, n))
                # check to ensure we stay positive
                if (np.sum(s[:, i+1] > 0) != n):
                    break
            if (i == t - 2):
                done = True
                break
    return [A, s, c]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 20
    t = 150
    sigma = 0.1
    A, s, C = generateVAR1(n, t, sigma)
    plt.plot(s)
    print(C)