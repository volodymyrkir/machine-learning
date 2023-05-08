from functools import reduce

import pandas as pd

from utils import discretize_info_gain, split_train_test
from utils import replace_nulls, shuffle_dataframe


class NaiveBayesManager:

    def __init__(self, train_ranges):
        self._datasets = {}
        self._train_ranges = train_ranges

    def import_dataset(self,
                       data: pd.DataFrame,
                       dataframe_name: str,
                       target_col: str,
                       discretize: bool = False,
                       contains_nulls: bool = False,
                       default_null: str = '?') -> None:
        if contains_nulls:
            data = replace_nulls(data, target_col, default_null)
        if discretize:
            data = discretize_info_gain(data, target_col)
        data = shuffle_dataframe(data)
        self._datasets[dataframe_name] = data, target_col

    def collect_results(self) -> pd.DataFrame:
        data = []
        for df_name, df_payload in self._datasets.items():
            df, target_col = df_payload
            for train_percent in self._train_ranges:
                predicted, actual = self._apply_bayes_algorithm(
                    df,
                    train_percent,
                    target_col
                )
                accuracy = self._calculate_accuracy(predicted, actual)
                data.append([df_name, f'{train_percent}%', accuracy])

        result = pd.DataFrame(data, columns=['DatasetName', 'TrainPercent', 'Accuracy'])
        print(result)
        return result

    @staticmethod
    def _apply_bayes_algorithm(data: pd.DataFrame,
                               train_percent: int,
                               target_col: str):

        train, test = split_train_test(data, round(train_percent / 100, 2))

        class_num = train[target_col].nunique()
        classes = train[target_col].unique().tolist()

        feature_cols = [col for col in data.columns if col != target_col]
        subsets = {cls: train.loc[train[target_col] == cls] for cls in classes}

        predicted_values = []
        for _, row in test.iterrows():
            class_probs = {cls: [] for cls in classes}

            for feature in feature_cols:
                feature_value = row[feature]

                for cls in classes:
                    subset = subsets[cls]
                    value_count = len(subset.loc[subset[feature] == feature_value])
                    class_probs[cls].append((value_count + 1) / (len(subset) + class_num))

            class_probs = {cls: reduce(lambda x, y: x*y, class_probs[cls]) for cls in class_probs}
            predicted_values.append(max(class_probs, key=class_probs.get))

        actual = test[target_col].tolist()

        return predicted_values, actual

    @staticmethod
    def _calculate_accuracy(predicted: list, actual: list):
        correct = 0
        total = len(actual)
        for case in range(len(actual)):
            if predicted[case] == actual[case]:
                correct += 1
        return 0 if correct == 0 else round(correct/total, 3)

    def test_func(self):
        for target, data in self._datasets.items():
            print(data)
