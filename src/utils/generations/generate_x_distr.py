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


def generate_x_values(x_occurrences, n):
    # calculate the total number of occurrences
    total_x = sum(x_occurrences.values())

    x_probabilities = calculate_probabilities(x_occurrences, total_x)

    # sort the list based on probabilities in ascending order
    x_probabilities.sort(key=lambda probability: probability[1])

    # generate the list of values given n
    x_values = get_rand_values(x_probabilities, n)

    return x_values


if __name__ == "__main__":
    x_occurrences = {"100": 1, "20": 1, "40": 1}
    generated_values = generate_x_values(x_occurrences, 10)
    print(generated_values)
