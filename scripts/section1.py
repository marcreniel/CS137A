#!/usr/bin/env python3

# add import and helper functions here
import numpy as np

if __name__ == "__main__":
    print("hello world")
    np.random.seed(42)
    A = np.random.normal(size=(4, 4))
    B = np.random.normal(size=(4, 2))
    dot = A @ B
    print(dot)