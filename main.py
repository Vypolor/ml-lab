from turtle import st

import numpy as np
import scipy.stats as st
from utils.reader import read_data_from_file

DATA_PATH = "data/data.txt"
DATA_TRUE_PATH = "data/data_true.txt"
DATA_FAKE_PATH = "data/data_fake.txt"
SEPARATOR = ";"
WINDOW_SIZE = 8
ALPHA = 0.99999999999999


def transition_matrix(data, states):
    matrix = np.full((len(states), len(states)), 0.1)
    for i in range(len(data) - 1):
        matrix[states.index(data[i]), states.index(data[i + 1])] += 1
    for i in range(len(states)):
        matrix[i] = matrix[i] / matrix[i].sum()
    return matrix


def get_probability(data, matrix, states):
    prob = 1
    for i in range(len(data) - 1):
        prob *= matrix[states.index(data[i]), states.index(data[i + 1])]
    return prob


def get_states(data):
    result = []
    for i, row in data.iterrows():
        result += row.Data.split(SEPARATOR)
    return list(set(result))


def get_probabilities(data, matrix, states):
    probs = []
    if len(data) > WINDOW_SIZE:
        for i in range(len(data) - WINDOW_SIZE):
            win_data = data[i: i + WINDOW_SIZE + 1]
            probs.append(get_probability(win_data, matrix, states))
    else:
        probs.append(get_probability(data, matrix, states))

    return probs


def anomalies_checking(data, matrix, interval, states):
    probs = get_probabilities(data, matrix, states)
    for prob in probs:
        if not (interval[0] < prob < interval[1]):
            return 1
    return 0


def get_confidence_interval(data, matrix, states):
    probabilities = np.array(get_probabilities(data, matrix, states))
    interval = st.t.interval(ALPHA, df=len(probabilities) - 1, loc=np.median(probabilities), scale=st.sem(probabilities))
    if interval[0] < 0:
        interval = (min(probabilities), interval[1])
    if interval[1] > 1:
        interval = (interval[0], max(probabilities))
    if np.isnan(interval[0]) and np.isnan(interval[1]):
        interval = (np.min(probabilities), np.max(probabilities))

    return interval


def get_result_values(i, row):
    data1 = row.Data.split(SEPARATOR)
    data_true1 = data_true_map.Data[i].split(SEPARATOR)
    data_fake1 = data_fake_map.Data[i].split(SEPARATOR)
    matrix = transition_matrix(data1, states_values)
    interval = get_confidence_interval(data1, matrix, states_values)
    true_result_array.append(anomalies_checking(data_true1, matrix, interval, states_values))
    fake_result_array.append(anomalies_checking(data_fake1, matrix, interval, states_values))


if __name__ == '__main__':
    data_map = read_data_from_file(DATA_PATH)
    data_true_map = read_data_from_file(DATA_TRUE_PATH)
    data_fake_map = read_data_from_file(DATA_FAKE_PATH)

    states_values = get_states(data_map)

    true_result_array = []
    fake_result_array = []
    interval = ()

    for i, row in data_map.iterrows():
        get_result_values(i, row)

    print("Data true: ", (1 - (sum(true_result_array) / data_true_map.shape[0])) * 100, "%")
    print("Data fake: ", (sum(fake_result_array) / data_fake_map.shape[0]) * 100, "%")
