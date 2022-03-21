import numpy as np

# GRADED FUNCTION

# This is the Gaussian function.
def f(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig


# Next up, the derivative with respect to μ.
# If you wish, you may want to express this as f(x, mu, sig) multiplied by chain rule terms.
# === COMPLETE THIS FUNCTION ===
def dfdmu(x, mu, sig):
    return f(x, mu, sig) * (x - mu) / (sig**2)


# Finally in this cell, the derivative with respect to σ.
# === COMPLETE THIS FUNCTION ===
def dfdsig(x, mu, sig):
    return f(x, mu, sig) * ((((x - mu) ** 2) / sig**3) - (1 / sig))


# GRADED FUNCTION

# Complete the expression for the Jacobian, the first term is done for you.
# Implement the second.
# === COMPLETE THIS FUNCTION ===
def steepest_step(x, y, mu, sig, aggression):
    J = np.array(
        [
            -2 * (y - f(x, mu, sig)) @ dfdmu(x, mu, sig),
            -2 * (y - f(x, mu, sig)) @ dfdsig(x, mu, sig),
        ]
    )
    step = -J * aggression
    return step


# Test your code before submissio
"""To test the code you've written above, run all previous cells (select each cell, then press the play button [ ▶| ] or press shift-enter). 
You can then use the code below to test out your function. 
You don't need to submit these cells;
You can edit and run them as much as you like."""
