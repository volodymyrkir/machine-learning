import itertools
from typing import Callable
from collections import Counter

import pandas as pd

from k_nearest_neighbors.knn_utils import simple_knn

from utils import replace_nulls, shuffle_dataframe, split_train_test, split_train_validation_test
from ensemble_utils import cross_validation, bagging

K = 3
N = 10


class Ensembler:

    def __init__(self):
        self._datasets = {}
        self.results = {
            'Dataset': [],
            'DataSelectionMethod': [],
            'Voting': [],
            'Accuracy': [],
        }

    def import_dataset(self,
                       data: pd.DataFrame,
                       dataframe_name: str,
                       target_col: str,
                       contains_nulls: bool = False,
                       default_null: str = '?') -> None:
        if contains_nulls:
            data = replace_nulls(data, target_col, default_null)
        data = shuffle_dataframe(data).reset_index(drop=True)

        self._datasets[dataframe_name] = data, target_col

    @staticmethod
    def _predict_ensemble(y: pd.Series,
                          target_col: str,
                          n_classifiers: int,
                          data_slices) -> dict:
        predictions = {}
        for predictor in range(n_classifiers):
            predictions[predictor] = simple_knn(data_slices[predictor], target_col, K, y)
        return predictions

    @staticmethod
    def _simple_knn(data: pd.DataFrame,
                    target_col: str) -> float:
        """
        Predicts and calculates accuracy
        :param data: dataframe
        :param target_col: target dataframe`s col

        :return: accuracy of algorithm
        """
        predictions = []
        data = shuffle_dataframe(data)
        train, test = split_train_test(data)
        test_dropped = test.drop(columns=[target_col])
        for row in range(len(test)):
            instance = test_dropped.iloc[row]
            prediction = simple_knn(train, target_col, K, instance)
            actual = test.iloc[row][target_col]
            predictions.append(prediction == actual)

        accuracy = round((sum(predictions) / len(test)) * 100, 2)

        return accuracy

    def _general_voting(self,
                        data: pd.DataFrame,
                        target_col: str,
                        data_pick_method: Callable) -> float:
        result_predictions = []
        train, test = split_train_test(data)
        test_dropped = test.drop(columns=[target_col])

        data_slices = data_pick_method(train, N)
        for row in range(len(test_dropped)):
            instance = test_dropped.iloc[row]
            ensemble_voting = self._predict_ensemble(instance, target_col, N, data_slices)
            predictions = Counter(ensemble_voting.values())
            actual = test.iloc[row][target_col]
            result_predictions.append(max(predictions, key=predictions.get) == actual)

        accuracy = round((sum(result_predictions) / len(test)) * 100, 2)

        return accuracy

    def _static_weighted_voting(self,
                                data: pd.DataFrame,
                                target_col: str,
                                data_pick_method: Callable) -> float:
        result_predictions = []
        train, validate, test = split_train_validation_test(data)
        test_dropped = test.drop(columns=[target_col])
        validate_dropped = validate.drop(columns=[target_col])
        data_slices = data_pick_method(train, N)
        classifiers = {i: 0 for i in range(N)}

        for row in range(len(validate_dropped)):
            instance = validate_dropped.iloc[row]
            ensemble_voting = self._predict_ensemble(instance, target_col, N, data_slices)
            actual = validate.iloc[row][target_col]
            classifiers = {
                i: classifiers[i] + 1 if actual == ensemble_voting[i]
                else classifiers[i]
                for i in ensemble_voting
            }
        max_classifiers = max(classifiers.values())
        classifiers = {key: value / max_classifiers for key, value in classifiers.items()}

        for row in range(len(test_dropped)):
            instance = test_dropped.iloc[row]
            ensemble_voting = self._predict_ensemble(instance, target_col, N, data_slices)
            class_votings = {value: 0 for value in train[target_col].unique().tolist()}
            for classifier, weight in classifiers.items():
                class_votings[ensemble_voting[classifier]] += weight

            actual = test.iloc[row][target_col]
            result_predictions.append(max(class_votings, key=class_votings.get) == actual)

        accuracy = round((sum(result_predictions) / len(test)) * 100, 2)

        return accuracy

    def _weighted_majority_voting(self,
                                  data: pd.DataFrame,
                                  target_col: str,
                                  data_pick_method: Callable) -> float:
        result_predictions = []
        train, validate, test = split_train_validation_test(data)
        test_dropped = test.drop(columns=[target_col])
        validate_dropped = validate.drop(columns=[target_col])
        data_slices = data_pick_method(train, N)
        classifiers = {i: 1 for i in range(N)}

        for row in range(len(validate_dropped)):
            instance = validate_dropped.iloc[row]
            ensemble_voting = self._predict_ensemble(instance, target_col, N, data_slices)
            actual = validate.iloc[row][target_col]
            for classifier in classifiers:
                if ensemble_voting[classifier] != actual:
                    classifiers[classifier] *= 0.5

        max_classifiers = max(classifiers.values())
        classifiers = {key: value / max_classifiers for key, value in classifiers.items()}

        for row in range(len(test_dropped)):
            instance = test_dropped.iloc[row]
            ensemble_voting = self._predict_ensemble(instance, target_col, N, data_slices)
            class_votings = {value: 0 for value in train[target_col].unique().tolist()}
            for classifier, weight in classifiers.items():
                class_votings[ensemble_voting[classifier]] += weight

            actual = test.iloc[row][target_col]
            result_predictions.append(max(class_votings, key=class_votings.get) == actual)

        accuracy = round((sum(result_predictions) / len(test)) * 100, 2)

        return accuracy

    def _collect_results(self):

        for dataset_name, data in self._datasets.items():
            accuracy = self._simple_knn(*data)
            self.results['Dataset'].append(dataset_name),
            self.results['DataSelectionMethod'].append(None),
            self.results['Voting'].append(self._simple_knn.__name__[1:]),
            self.results['Accuracy'].append(f'{accuracy}%'),
            print(f'accuracy running {self._simple_knn.__name__[1:]} '
                  f'for {dataset_name} dataset, '
                  f'using None data pick method '
                  f'is {accuracy}%')

        for data, data_pick_method, algorithm in \
                itertools.product(
                    self._datasets.items(),
                    [cross_validation, bagging],
                    [self._general_voting, self._static_weighted_voting, self._weighted_majority_voting]
                ):

            dataset_name, df_data = data

            accuracy = algorithm(
                *df_data,
                data_pick_method
            )

            self.results['Dataset'].append(dataset_name),
            self.results['DataSelectionMethod'].append(data_pick_method.__name__),
            self.results['Voting'].append(algorithm.__name__[1:]),
            self.results['Accuracy'].append(f'{accuracy}%'),
            print(f'accuracy running {algorithm.__name__[1:]} '
                  f'for {dataset_name} dataset, '
                  f'using {data_pick_method.__name__} data pick method '
                  f'is {accuracy}%')

    def print_result_table(self):
        self._collect_results()
        results_df = pd.DataFrame.from_dict(self.results)
        print('\n', results_df)
