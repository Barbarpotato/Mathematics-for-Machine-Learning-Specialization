import numpy as np

# Shape Exxample 1
L = np.array(
    [
        [0, 1 / 2, 1 / 3, 0, 0, 0],
        [1 / 3, 0, 0, 0, 1 / 2, 0],
        [1 / 3, 1 / 2, 0, 1, 0, 1 / 2],
        [1 / 3, 0, 1 / 3, 0, 1 / 2, 1 / 2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1 / 3, 0, 0, 0],
    ]
)

# Shape Exxample 2
L2 = np.array(
    [
        [0, 1 / 2, 1 / 3, 0, 0, 0, 0],
        [1 / 3, 0, 0, 0, 1 / 2, 0, 0],
        [1 / 3, 1 / 2, 0, 1, 0, 1 / 3, 0],
        [1 / 3, 0, 1 / 3, 0, 1 / 2, 1 / 3, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1 / 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1 / 3, 1],
    ]
)

# Create Basic PageRank
def create_Basic_PageRank(shapes):
    copy_shapes = np.array(shapes, dtype=float)
    n_Matrix = shapes.shape[0]
    r = 100 * np.ones(n_Matrix, dtype=float) / n_Matrix
    r = copy_shapes @ r

    for i in range(0, 100):  # stabilizing our vector r
        copy_r = np.array(r, dtype=float)
        r = copy_shapes @ r
        if np.array_equal(copy_r, r) == True:
            return r


# pagerank = create_Basic_PageRank(L)
# print(pagerank)

# Create Dumping_Page Rank
def create_Dumping_PageRank(shapes, d=0.5):
    copy_shapes = np.array(shapes, dtype=float)
    n_Matrix = shapes.shape[0]
    M = d * copy_shapes + (1 - d) / n_Matrix * np.ones(
        [n_Matrix, n_Matrix], dtype=float
    )
    r = 100 * np.ones(n_Matrix, dtype=float) / n_Matrix

    for i in range(0, 100):  # stabilizing our vector r
        copy_r = np.array(r, dtype=float)
        r = M @ r
        if np.array_equal(copy_r, r) == True:
            return r


pagerank = create_Dumping_PageRank(L)
print(pagerank)

# Sorting PageRank from the biggest to the lowest.
def sortPageRank(rank):
    for i in range(len(rank)):
        for j in range(i + 1, len(rank)):
            if rank[j] > rank[i]:
                max = rank[j]
                rank[i], rank[j] = max, rank[i]

    for i in range(len(rank)):
        print("rank:{} , probabilities:{}".format(i + 1, rank[i]))


# sortPageRank(r)
