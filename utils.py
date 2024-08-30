import numpy as np


def generate_markov_chain_samples(
    initial_distribution, transition_matrix, num_samples, sequence_length
):
    # Ensure initial_distribution and transition_matrix are numpy arrays
    initial_distribution = np.array(initial_distribution)
    transition_matrix = np.array(transition_matrix)

    # Get the number of states
    num_states = len(initial_distribution)

    # Initialize the tensor to store the samples
    samples = np.zeros((num_samples, sequence_length), dtype=int)

    for i in range(num_samples):
        # Sample the initial state
        samples[i, 0] = np.random.choice(num_states, p=initial_distribution)

        for j in range(1, sequence_length):
            # Sample the next state based on the current state and transition matrix
            samples[i, j] = np.random.choice(
                num_states, p=transition_matrix[samples[i, j - 1]]
            )

    return samples


def gen_tran_mat(vocab_size, sig=1, sparsity=None):
    mat = np.exp(sig * np.random.random((vocab_size, vocab_size)))
    mat = mat / mat.sum(axis=-1)[:, None]
    if sparsity is not None:
        cutoff = np.quantile(mat.flatten(), 1 - sparsity)
        mat[mat < cutoff] = 0
        mat = mat / mat.sum(axis=-1)[:, None]
    return mat


def pca(X, k):
    X = X - np.mean(X, axis=0)
    u, s, vt = np.linalg.svd(X)

    return X @ vt[:k, :].T
