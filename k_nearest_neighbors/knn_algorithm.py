from typing import Callable

import pandas as pd

from utils import split_train_test


class KNNManager:

    def __init__(self, dataset_dict: dict,
                 k: list[int],
                 algorithms: list[Callable]):

        self._datasets = dataset_dict
        self._ks = k
        self._algorithms = algorithms

    @staticmethod
    def _predict(algorithm: Callable,
                neighbors: int,
                dataset_name: str,
                data: pd.DataFrame,
                target_col: str,
                attributes_range: slice
                ) -> float:
        """
        Predicts and calculates accuracy
        :param data: dataframe
        :param algorithm: function that is creating k-nn prediction
        :param target_col: target dataframe`s col
        :param attributes_range: slice object ( slice(from,to,step) )
        :param neighbors: num of neighbors
        :param dataset_name: name of the dataset for logging
        :return: accuracy of algorithm
        """
        predictions = []
        train, test = split_train_test(data, 0.7)
        for row in range(len(test)):
            instance = test.iloc[row, attributes_range]
            prediction = algorithm(train, target_col, neighbors, instance)
            actual = test.iloc[row][target_col]
            predictions.append(prediction == actual)

        accuracy = round(sum(predictions) / len(test), 3)

        print(f'accuracy running {algorithm.__name__} '
              f'for {dataset_name} dataset '
              f'with {neighbors} neighbors is {accuracy}')
        return accuracy

    def collect_results(self):
        results = {}
        for name, values in self._datasets.items():
            for algorithm in self._algorithms:
                for k in self._ks:
                    results[(name, k, algorithm)] = []

        for name, values in self._datasets.items():
            for k in self._ks:
                for algorithm in self._algorithms:
                    results[(name, k, algorithm)].append(
                        self._predict(algorithm,  k, name, *values))
        print(results)
