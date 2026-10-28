import numpy as np

# s(t+1)=A*s(t)+sigma*Z(t)
# Estimate the matrix A above from a given time series s
# using Moore-Penrose pseudoinverse
# @author R. Sipos


def est_A_pinv(s):
    dim = s.shape
    Tw = dim[1]-1
    # below works on s'
    A = np.zeros((dim(0), dim(0)))

    An_coeff = np.zeros((dim(0), dim(0)))
    An_coeff[1:dim[0], 1:dim[0]] = s[dim[0]-Tw:dim[0]-1, 1:dim[0]] * s[dim[0]-Tw:dim[0]-1, 1:dim[0]]

    An_inv = np.zeros(dim[0], dim[0])  # ok<NASGU>
    if np.linalg.matrix_rank(An_coeff) == dim:
        An_inv = np.invert(An_coeff)
    else:
        An_inv = np.linalg.pinv(An_coeff) # Moore-Penrose pseudoinverse

    for n in range(dim):
        b = np.zeros(dim[0], 1)
        b[range(dim[0])] = s[dim[1]-Tw+1:dim[1], n] * s[dim[1]-Tw:dim[1]-1, 1:dim[0]]
        A[n, :] = An_inv * b

    print('done')

    return A


def estA(s):
    # s(t+1)=A*s(t)+sigma*Z(t)
    # Estimate the matrix A above from a given time series s
    # using Moore-Penrose pseudoinverse
    dim = s.shape
    # Estimation method for A depends on n>=t vs t<n
    if dim[0] >= dim[1]:
        # underdetermined case for A, we can solve st=A*st-1 exactly
        st1 = s[:, range(dim[1] - 1)]
        st = s[:, range(1, dim[1])]
        A = np.matmul(st, np.linalg.inv(st1))
    else:
        # Compute A_hat=(s^T_t-1*s_t-1)^(-1)*(s^T_t-1*S_t)
        B = np.zeros((dim[0], dim[0]))
        C = np.zeros((dim[0], dim[0]))
        for i in range(dim[0]):
            for j in range(dim[0]):
                for k in range(1, dim[1]):
                    B[i, j] += s[i, k - 1] * s[j, k - 1]
                    C[i, j] += s[i, k] * s[j, k - 1]

        # should be pinv(B)*C, not clear why below works...
        # probably i and j above are mixed up somehow.
        A = np.matmul(C, np.linalg.pinv(B))
    return A