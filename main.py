import numpy as np

vectors = [
    np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]),
    np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
    np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
]

def generate_noisy_vectors(vector: np.ndarray) -> list[np.ndarray]:
    noisy_list = []
    noisy = vector.copy()
    for i in range(len(noisy)):
        noisy[i] = 1 - noisy[i]
        noisy_list.append(noisy.copy())
    return noisy_list

def sign(vector: np.ndarray) -> np.ndarray:
    return np.where(vector > 0, 1, 0)

def create_hopfield_weight(target_vector:np.ndarray) -> np.ndarray:
    target = target_vector.copy().reshape(1, -1)
    weight = (2 * target - 1).T * (2 * target - 1)
    I = np.eye(weight.shape[0], weight.shape[1])
    weight = weight - I
    return weight

def asyncHopfield(target_vector: np.ndarray, noisy_vector: np.ndarray) -> np.ndarray:
    noisy = noisy_vector.copy().reshape(1, -1)
    weight = create_hopfield_weight(target_vector)
    prev = noisy.copy()
    while True:
        for i in range(noisy.shape[1]):
            S = np.dot(noisy, weight[:, i])
            noisy[0, i] = sign(S)[0]
        if np.array_equal(noisy, prev):
            break
        prev = noisy.copy()
    return noisy.flatten()

def syncHopfield(target_vector: np.ndarray, noisy_vector: np.ndarray) -> np.ndarray:
    noisy = noisy_vector.copy().reshape(1, -1)
    weight = create_hopfield_weight(target_vector)
    prev = noisy.copy()
    while True:
        S = np.dot(noisy, weight)
        noisy = sign(S)
        if np.array_equal(noisy, prev):
            break
        prev = noisy.copy()
    return noisy.flatten()

def bin_to_pm1(vector: np.ndarray) -> np.ndarray:
    return 2 * vector - 1

def pm1_to_bin(vector: np.ndarray) -> np.ndarray:
    return ((vector + 1) // 2).astype(int)

def hamming_work(init_activation:np.ndarray, e = 0.1 , max_iterations = 25) -> np.ndarray:
    z = init_activation.copy()
    m = len(z)
    weight = np.full((m,m),-e)
    np.fill_diagonal(weight,1)
    for _ in range(max_iterations):
        new_z = np.dot(weight, z)
        new_z = np.where(new_z > 0, new_z, 0)
        if np.sum(new_z > 0) == 1:
            return new_z
        z = new_z
    return z

def hamming_network(noisy_vector:np.ndarray, prototypes:np.ndarray) -> tuple[np.ndarray, int]:
    noisy_pm1 = bin_to_pm1(noisy_vector)
    prototypes_pm1 = bin_to_pm1(prototypes)

    weights = prototypes_pm1 / 2
    n = len(noisy_pm1)
    T = n / 2
    s = np.dot(weights, noisy_pm1) + T

    z_last = hamming_work(s)
    vector_winner = np.argmax(z_last)
    return z_last, vector_winner

def hamming_restoration(noisy_vector: np.ndarray, prototypes: np.ndarray) -> tuple[list[np.ndarray],int,np.ndarray]:
    z_last, vector_winner = hamming_network(noisy_vector, prototypes)
    # candidate_indices = np.where(z_last == np.max(z_last))[0] 
    candidate_indices = np.where((z_last == np.max(z_last)) & (np.max(z_last) > 0))[0]
    #print(candidate_indices, z_last)
    restored_vectors = []
    for idx in candidate_indices:
        restored_vectors.append(prototypes[idx])
    return restored_vectors, vector_winner, z_last

def test_hamming(vector: np.ndarray, prototypes: list[np.ndarray]) -> int:
    max_bits_hamming = 0
    noisy_versions = generate_noisy_vectors(vector)
    prototypes_array = np.array(prototypes)

    for noisy in noisy_versions:
        restored_vectors, winner_idx, activations = hamming_restoration(noisy, prototypes_array)
        best_score = -1
        best_restored = None

        for restored in restored_vectors:
            result = asyncHopfield(vector, restored)
            score = np.sum(result == vector)
            if score > best_score:
                best_score = score
                best_restored = result
        # print(best_restored, vector)
        if np.array_equal(best_restored, vector):
            max_bits_hamming += 1
        # else:
        #     break
    return max_bits_hamming

def test_asyncHopfield(vector: np.ndarray) -> int:
    max_bits_async = 0
    noisy_versions = generate_noisy_vectors(vector)
    for noisy in noisy_versions:
        recovered = asyncHopfield(vector, noisy)
        if np.array_equal(recovered, vector):
            max_bits_async += 1
        else:
            break
    return max_bits_async

def test_syncHopfield(vector: np.ndarray) -> int:
    max_bits_sync = 0
    noisy_versions = generate_noisy_vectors(vector)
    for noisy in noisy_versions:
        recovered = syncHopfield(vector, noisy)
        if np.array_equal(recovered, vector):
            max_bits_sync += 1
        else:
            break
    return max_bits_sync

def init_weight_bam(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.dot(x.T, y)

def bam_recover(x: np.ndarray, weight: np.ndarray, y_mode: bool) -> np.ndarray:
    if y_mode:
        return sign(np.dot(x, weight))
    else:
        return sign(np.dot(x, weight.T))
def test_bam_x_bits(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> int:
    max_x_bits = 0
    x_test = x.copy()
    for i in range(x.shape[1]):
        x_test[0, i] = 1 - x_test[0, i]
        x_test_signed = sign(x_test)
        y_recovered = bam_recover(x_test_signed, weight, True).flatten()
        if np.array_equal(y_recovered, y.flatten()):
            max_x_bits += 1
        else:
            break
    return max_x_bits

def test_bam_y_bits(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> int:
    max_y_bits = 0
    y_test = y.copy()
    for i in range(y.shape[1]):
        y_test[0, i] = 1 - y_test[0, i]
        y_test_signed = sign(y_test)
        x_recovered = bam_recover(y_test_signed, weight, False).flatten()
        if np.array_equal(x_recovered, x.flatten()):
            max_y_bits += 1
        else:
            break
    return max_y_bits

def test_bam(vector: np.ndarray) -> tuple[int, int]:
    x = vector[:10].reshape(1, -1)
    y = vector[10:].reshape(1, -1)
    weight = init_weight_bam(x, y)

    max_x_bits = test_bam_x_bits(x, y, weight)
    max_y_bits = test_bam_y_bits(x, y, weight)

    return max_x_bits, max_y_bits



if __name__ == "__main__":
    print("Recovery test results for each vector:\n")
    for i, vec in enumerate(vectors):
        async_recovered = test_asyncHopfield(vec)
        sync_recovered = test_syncHopfield(vec)
        max_x_bits, max_y_bits = test_bam(vec)
        hamming_recovered = test_hamming(vec, vectors)

        print(f"Vector {i + 1}:")
        print(f"  Maximum number of recoverable bits Hopfield network (Asynchronously): {async_recovered}")
        print(f"  Maximum number of recoverable bits Hopfield network (Synchronously): {sync_recovered}")
        print(f"  Maximum number of recoverable X bits (BAM): {max_x_bits}")
        print(f"  Maximum number of recoverable Y bits (BAM): {max_y_bits}")
        print(f"  Maximum number of recoverable bits (Hamming Network): {hamming_recovered}")
        print("-" * 50)