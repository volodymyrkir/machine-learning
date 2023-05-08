"""This module encapsulates utilities for all projects"""
from math import log
import os

import pandas as pd

IRIS_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
IRIS_ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
IRIS_TARGET = 'species'
IRIS_NAME = 'iris'

CONGRESS_COLS = ['class', *[f'c{n}' for n in range(1, 17)]]
CONGRESS_TARGET = 'class'
CONGRESS_ATTRIBUTES = [f'c{n}' for n in range(1, 17)]
CONGRESS_NAME = 'congress'

WINE_COLS = ["Class", "Alcohol", "Malic acid",
             "Ash", "Alcalinity of ash",
             "Magnesium", "Total phenols", "Flavanoids",
             "Nonflavanoid phenols", "Proanthocyanins",
             "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
WINE_TARGET = 'Class'
WINE_ATTRIBUTES = [attr for attr in WINE_COLS if attr != WINE_TARGET]
WINE_NAME = 'wine'

BREAST_TARGET = 'diagnosis'
BREAST_NAME = 'breast'


def get_data(name: str, cols=None) -> pd.DataFrame:
    return pd.read_csv(os.path.join(os.getcwd(), '..', 'input_data', name),
                       names=cols)


def split_train_test(dataframe: pd.DataFrame, fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a Pandas DataFrame into train and test sets.

    Args:
        dataframe: The Pandas DataFrame to split.
        fraction: The fraction of the data to use for the train set.

    Returns:
        A tuple containing the train and test DataFrames.
    """
    num_train = int(len(dataframe) * fraction)
    train_df = dataframe.iloc[:num_train].copy()
    test_df = dataframe.iloc[num_train:].copy()
    return train_df, test_df


def shuffle_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.sample(len(dataframe))


def get_column_max_gain(data_frame: pd.DataFrame, target_col: str, attributes: list[str]) -> str:
    """ Given data_frame, selects the column with the max info gain"""
    entropy_general = entropy(data_frame, target_col)
    instances = data_frame.shape[0]
    attributes_gains = {}
    for attribute in attributes:
        attr_value = entropy_general
        for value in data_frame[attribute].unique():
            new_frame = data_frame.loc[data_frame[attribute] == value]
            attr_value -= (new_frame.shape[0] / instances) * entropy(new_frame, target_col)
        attributes_gains[attribute] = attr_value
    return sorted(attributes_gains.items(),
                  key=lambda x: x[1],
                  reverse=True)[0][0]


def entropy(data: pd.DataFrame, target_col: str) -> float:
    """
    Calculates the entropy of a dataset based on the distribution of a target variable.

    Args:
        data (pd.DataFrame): The input dataset.
        target_col (str): The name of the target column.

    Returns:
        float: The entropy of the dataset.
    """
    counts = data[target_col].value_counts()
    total = counts.sum()
    entropy_value = 0
    for count in counts:
        prob = count / total
        entropy_value += -prob * log(prob, 2)
    return entropy_value


def information_gain_col(data: pd.DataFrame, target_col: str, feature_col: str) -> float:
    total_entropy = entropy(data, target_col)
    feature_values = data[feature_col].unique()
    feature_entropy = 0
    for value in feature_values:
        subset_indices = data.index[data[feature_col] == value]
        subset_data = data.loc[subset_indices]
        subset_entropy = entropy(subset_data, target_col)
        prob = len(subset_indices) / len(data)
        feature_entropy += prob * subset_entropy
    return total_entropy - feature_entropy


def replace_nulls(data_frame: pd.DataFrame, target_col: str, default_null='?'):
    """Replaces nulls in dataset using most common value in class """
    non_class_columns = [col for col in data_frame.columns if col != target_col]
    for column in non_class_columns:
        for value in data_frame[target_col].unique():
            most_common_value = data_frame[data_frame[target_col] == value][column].mode()[0]

            data_frame.loc[
                (data_frame[column] == default_null) &
                (data_frame[target_col] == value), column
            ] = most_common_value
    return data_frame


def discretize_info_gain(dataframe: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Discretizes all continuous attributes in dataframe based on info gain for each column
    :param dataframe: target_df
    :param target_col: target column of dataframe
    :returns: dataframe with discretized values
    """
    dataframe_copy = dataframe.copy()
    numeric_cols = dataframe_copy.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    num_bins = len(dataframe[target_col].unique()) - 1
    total_instances = dataframe_copy.shape[0]

    for col in numeric_cols:
        df = dataframe_copy[[col, target_col]].copy()
        df = df.sort_values([col, target_col], ascending=True)
        thresholds = set(df.loc[df[target_col].ne(df[target_col].shift())][col][1:].tolist())
        thresholds_mapping = {}
        if len(thresholds) == num_bins:
            pass
        for threshold in thresholds:
            left, right = df.loc[df[col] <= threshold], df.loc[df[col] > threshold]
            inf_gain = entropy(df, target_col) - ((len(left) / total_instances) * entropy(left, target_col)
                                                  + (len(right) / total_instances) * entropy(right, target_col))
            thresholds_mapping[threshold] = inf_gain
        best_thresholds = sorted(thresholds_mapping.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:num_bins]
        prev_threshold = min(df[col])
        max_threshold = max(df[col])
        for threshold, _ in best_thresholds:
            dataframe_copy.loc[(dataframe[col] < threshold) &
                               (dataframe[col] >= prev_threshold), col] = f'{prev_threshold}-{threshold}'
            prev_threshold = threshold
        dataframe_copy.loc[dataframe[col] >= prev_threshold, col] = f'{prev_threshold}-{max_threshold}'
    return dataframe_copy
