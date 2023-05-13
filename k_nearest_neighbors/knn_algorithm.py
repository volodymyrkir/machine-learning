from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt

from utils import split_train_test, get_columns_gain, shuffle_dataframe
from knn_utils import attribute_weighted_knn, combined_weighted_knn, distance_weighted_knn, simple_knn


pd.set_option('display.max_rows', 500)


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
        data = shuffle_dataframe(data)
        train, test = split_train_test(data, train_fraction)
        for row in range(len(test)):
            instance = test.iloc[row, attributes_range]
            if algorithm in [combined_weighted_knn, attribute_weighted_knn]:
                prediction = algorithm(train, target_col, neighbors, instance, self._igs[dataset_name])
            else:
                prediction = algorithm(train, target_col, neighbors, instance)
            actual = test.iloc[row][target_col]
            predictions.append(prediction == actual)

        accuracy = round((sum(predictions) / len(test)) * 100, 2)

        print(f'accuracy running {algorithm.__name__} '
              f'for {dataset_name} dataset, '
              f'using {train_fraction * 100}% of data for training, '
              f'with {neighbors} neighbors is {accuracy}%')

        return accuracy

    def _collect_results(self):
        results = {}
        for name, values in self._datasets.items():
            results[name] = {}
            for algorithm in self._algorithms:
                results[name][algorithm.__name__] = {}
                for k in self._ks:
                    results[name][algorithm.__name__][k] = {}
                    for train in self._train_range:
                        results[name][algorithm.__name__][k][train] = {}

        for name, values in self._datasets.items():

            for algorithm in self._algorithms:
                for k in self._ks:
                    for train in self._train_range:
                        results[name][algorithm.__name__][k][train] = (self._predict(algorithm,
                                                                                     k,
                                                                                     train,
                                                                                     name,
                                                                                     *values))
        return results

    def process_results(self):
        results = self._collect_results()
        self.plot_train_neighbours('iris', simple_knn.__name__, results)
        self.plot_train_neighbours('wine', distance_weighted_knn.__name__, results)
        self.plot_train_neighbours('wine', attribute_weighted_knn.__name__, results)
        self.plot_train_neighbours('congress', simple_knn.__name__, results)
        self.plot_train_neighbours('congress', combined_weighted_knn.__name__, results)

        self.plot_train_classifier('iris', 3, results)
        self.plot_train_classifier('wine', 3, results)
        self.plot_train_classifier('congress', 3, results)

        return self.get_dataframes(results)

    def plot_train_neighbours(self, name, algorithm, results):
        train_values = [value * 100 for value in self._train_range]

        for y, style in zip(self._ks, ['-', '--', ':', '-.', (0, (5, 10)), (0, (3, 5, 1, 5))]):
            plt.plot(train_values, results[name][algorithm][y].values(),
                     label=f'{y}-Nearest Neighbors', linestyle=style)

        plt.xlabel('Train data size, %')
        plt.ylabel('Accuracy')
        plt.title(f'{algorithm} algorithm applied on {name} dataset with different NNs')
        plt.legend()
        plt.show()

    def plot_train_classifier(self, name, k, results):
        train_values = [value * 100 for value in self._train_range]

        for classifier, style in zip(self._algorithms, ['-', '--', ':', '-.']):
            plt.plot(train_values, results[name][classifier.__name__][k].values(),
                     label=f'{classifier.__name__}', linestyle=style)

        plt.xlabel('Train data size, %')
        plt.ylabel('Accuracy')
        plt.title(f'Classifiers applied on {name} dataset with k=3')
        plt.legend()
        plt.show()

    def get_dataframes(self, result: dict):
        dfs = {}
        for name, _ in self._datasets.items():
            rows_list = []
            for algorithm in result[name]:
                for k in result[name][algorithm]:
                    for train in result[name][algorithm][k]:
                        rows_list.append({
                            'algorithm': algorithm,
                            'k': k,
                            'train': train * 100,
                            'accuracy': result[name][algorithm][k][train]
                        })
            dfs[name] = pd.DataFrame(rows_list)

        for name, df in dfs.items():
            print(f"Result DataFrame for '{name}':")
            print(df)
