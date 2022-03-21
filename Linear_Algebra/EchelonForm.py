import numpy as np


def isSingular(a):
    B = np.array(
        a, dtype=np.float_
    )  # Make B as a copy of A, since we're going to alter it's values.
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False


# This next line defines our error flag. For when things go wrong if the matrix is singular.
class MatrixIsSingular(Exception):
    pass


# if A[0,0] = 0, maka baris 1 ditmbh dgn baris2, lalu di bagi dengan nilai baris 1
# utk dapat nilai 1 di A[0,0]
def fixRowZero(A):
    if A[0, 0] == 0:
        A[0] = A[0] + A[1]
    if A[0, 0] == 0:
        raise MatrixIsSingular()
    A[0] = A[0] / A[0, 0]
    return A


# A[1,0] must be zero,
# A[1,1] must be 1
def fixRowOne(A):
    A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        raise MatrixIsSingular()
    A[1] = A[1] / A[1, 1]
    return A


# A[2,0],A[2,1] must be zero
# A[2,2] must be 1
def fixRowTwo(A):
    # Insert code below to set the sub-diagonal elements of row two to zero (there are two of them).
    A[2] = A[2] - A[2, 0] * A[0]
    A[2] = A[2] - A[2, 1] * A[1]
    # Next we'll test that the diagonal element is not zero.
    if A[2, 2] == 0:
        # Insert code below that adds a lower row to row 2.
        A[2] = A[2] + A[3]
        A[2] = A[2] - A[2, 0] * A[0]
        A[2] = A[2] - A[2, 1] * A[1]
    # Now repeat your code which sets the sub-diagonal elements to zero.
    if A[2, 2] == 0:
        raise MatrixIsSingular()
    # Finally set the diagonal element to one by dividing the whole row by that element.
    A[2] = A[2] / A[2, 2]
    print(A[2])
    return A


# You should also complete this function
# Follow the instructions inside the function at each comment.
def fixRowThree(A):
    # Insert code below to set the sub-diagonal elements of row three to zero.
    A[3] = A[3] - A[3, 0] * A[0]
    A[3] = A[3] - A[3, 1] * A[1]
    A[3] = A[3] - A[3, 2] * A[2]
    # Complete the if statement to test if the diagonal element is zero.
    if A[3, 3] == 0:
        raise MatrixIsSingular()
    A[3] = A[3] / A[3, 3]
    # Transform the row to set the diagonal element to one.
    return A


test = np.array(
    [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 4], [0, 0, 5, 5]], dtype=np.float_
)
print(isSingular(test))

A = np.array(
    [[0, 7, -5, 3], [2, 8, 0, 4], [3, 12, 0, 5], [1, 3, 1, 3]], dtype=np.float_
)
fixRowZero(A)
fixRowOne(A)
fixRowTwo(A)
print(fixRowThree(A))
