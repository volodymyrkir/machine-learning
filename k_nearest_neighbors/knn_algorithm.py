from typing import Callable

import pandas as pd

from utils import split_train_test, get_columns_gain
from knn_utils import attribute_weighted_knn, combined_weighted_knn


class KNNManager:

    def __init__(self, dataset_dict: dict,
                 k: list[int],
                 algorithms: list[Callable],
                 train_range: list[float]):

        self._datasets = dataset_dict
        self._ks = k
        self._algorithms = algorithms
        self._train_range = train_range
        self._igs = self._get_igs()

    def _get_igs(self):
        igs = {}
        for name, dataset in self._datasets.items():
            ig_cols = get_columns_gain(dataset[0], dataset[1], [column
                                                                for column in dataset[0].columns
                                                                if column != dataset[1]])
            max_value = max(ig_cols.values())

            igs[name] = {key: value / max_value
                         for key, value in ig_cols.items()}
        return igs

    def _predict(self,
                 algorithm: Callable,
                 neighbors: int,
                 train_fraction: float,
                 dataset_name: str,
                 data: pd.DataFrame,
                 target_col: str,
                 attributes_range: slice
                 ) -> float:
        """
        Predicts and calculates accuracy
        :param data: dataframe
        :param train_fraction: fraction of data that is used for training
        :param algorithm: function that is creating k-nn prediction
        :param target_col: target dataframe`s col
        :param attributes_range: slice object ( slice(from,to,step) )
        :param neighbors: num of neighbors
        :param dataset_name: name of the dataset for logging
        :return: accuracy of algorithm
        """
        predictions = []
        train, test = split_train_test(data, train_fraction)
        for row in range(len(test)):
            instance = test.iloc[row, attributes_range]
            if algorithm in [combined_weighted_knn, attribute_weighted_knn]:
                prediction = algorithm(train, target_col, neighbors, instance, self._igs[dataset_name])
            else:
                prediction = algorithm(train, target_col, neighbors, instance)
            actual = test.iloc[row][target_col]
            predictions.append(prediction == actual)

        accuracy = round(sum(predictions) / len(test), 3)

        print(f'accuracy running {algorithm.__name__} '
              f'for {dataset_name} dataset, '
              f'using {train_fraction * 100}% of data for training, '
              f'with {neighbors} neighbors is {accuracy}')
        return accuracy

    def _collect_results(self):
        results = {}
        for name, values in self._datasets.items():
            results[name] = {}
            for algorithm in self._algorithms:
                results[name][algorithm] = {}
                for train in self._train_range:
                    results[name][algorithm][train] = {}
                    for k in self._ks:
                        results[name][algorithm][train][k] = {}

        for name, values in self._datasets.items():

            for algorithm in self._algorithms:
                for train in self._train_range:
                    for k in self._ks:
                        results[name][algorithm][train][k] = (self._predict(algorithm,
                                                                            k,
                                                                            train,
                                                                            name,
                                                                            *values))
        return results

    def plot_results(self, name='iris', algorithm='simple_knn'):
        results = self._collect_results()
        # train_values = list(self._train_range)
        # k_values = list(self._ks)
        # fig, ax = plt.subplots(len(k_values), len(train_values), figsize=(10, 10))
        #
        # for i, k in enumerate(k_values):
        #     for j, train in enumerate(train_values):
        #         ax[i, j].plot(train_values, results[name][algorithm][train][k])
        #         ax[i, j].set_title(f'k = {k}, train = {train}')
        #         ax[i, j].set_xlabel('Train range')
        #         ax[i, j].set_ylabel('Accuracy')
        #
        # plt.tight_layout()
        # plt.show()
