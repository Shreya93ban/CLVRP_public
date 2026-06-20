import os
import numpy as np
import pickle


def generate_dataset(seed=None):
    if seed is not None:
        np.random.seed(seed)
    customers = np.random.uniform(0, 100, size=(12, 2))
    return customers


def main():
    rng = np.random.default_rng(42)
    for i in range(0, 1):
        seed = int(rng.integers(0, 2**31 - 1))
        data_set = generate_dataset(seed)
        print(f"Generated dataset {i + 1} with seed {seed}:",'\n', data_set)


if __name__ == '__main__':
    main()
