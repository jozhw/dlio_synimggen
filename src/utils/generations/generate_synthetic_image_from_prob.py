import random

# first must get the total occurrances for each intensity value and entropy
# issue with entropy is that the numbers are particular. One option is
# to round to the nearest tenth.


def calculate_probabilities(occurrences, total):
    # create a list of tuples with value and probability
    probabilities = []
    for value, count in occurrences.items():
        probability = count / total
        probabilities.append((value, probability))

    return probabilities


def get_rand_values(probabilities, n):
    # generate the list of values given n
    values = []
    for _ in range(n):
        # generate a random number between 0 and 1 to be used to determine values
        random_number = random.random()

        # find the value based on the cumulative probabilities compared to random number
        cumulative_probability = 0
        for value, probability in probabilities:
            cumulative_probability += probability
            if random_number <= cumulative_probability:
                values.append(value)
                break

    return values


def generate_values(entropy_occurrences, mean_occurrences, dim_occurrences, n):
    # calculate the total number of occurrences
    total_entropy = sum(entropy_occurrences.values())
    total_mean = sum(mean_occurrences.values())
    total_dim = sum(dim_occurrences.values())

    entropy_probabilities = calculate_probabilities(entropy_occurrences, total_entropy)
    mean_probabilities = calculate_probabilities(mean_occurrences, total_mean)
    dim_probabilities = calculate_probabilities(dim_occurrences, total_dim)

    # sort the list based on probabilities in ascending order
    entropy_probabilities.sort(key=lambda probability: probability[1])
    mean_probabilities.sort(key=lambda probability: probability[1])
    dim_probabilities.sort(key=lambda probability: probability[1])

    # generate the list of values given n
    entropy_values = get_rand_values(entropy_probabilities, n)
    mean_values = get_rand_values(mean_probabilities, n)
    dim_values = get_rand_values(dim_probabilities, n)

    values = []

    for i in range(n):
        values.append((entropy_values[i], mean_values[i], dim_values[i]))

    return values


if __name__ == "__main__":
    entropy_occurrences = {"1.5": 1, "4": 1, "7": 1}
    mean_occurrences = {"0": 3, "100": 2, "255": 5}
    dim_occurrences = {(1, 1, 1): 1, (2, 2, 2): 2, (3, 3, 3): 1}
    generated_values = generate_values(
        entropy_occurrences, mean_occurrences, dim_occurrences, 10
    )
    print(generated_values)
