import pandas as pd
import numpy as np

from utils import information_gain_col

from utils import CONGRESS_ATTRIBUTES


def prepare_numeric(dataframe: pd.DataFrame,
                    numeric_cols: list[str]) -> pd.DataFrame:
    """
    Brings all numeric values to 0-1 scale
    :param numeric_cols: list of numeric cols
    :param dataframe: dataframe
    :return: dataframe with cols converted to 0-1 scale
    """
    dataframe.loc[:, numeric_cols] = (dataframe[numeric_cols].apply(
        lambda col: col - min(col) if min(col) < 0 else col)
    )
    dataframe.loc[:, numeric_cols] = (dataframe[numeric_cols].apply(
        lambda col: col / max(col) if max(col) != 0 else 0)
    )
    return dataframe


def prepare_categorical(dataframe: pd.DataFrame,
                        categorical_cols: list[str] = CONGRESS_ATTRIBUTES) -> pd.DataFrame:
    """
    Brings y-n values to 0-1 values
    :param categorical_cols: list of categorical column names
    :param dataframe: target dataframe
    :return: dataframe with categorical values replaced by 0-1 values
    """
    for col in categorical_cols:
        dataframe[col] = dataframe[col].replace({'y': 1, 'n': 0})
    return dataframe


def simple_knn(dataframe: pd.DataFrame, target_col: str, k: int, y: pd.Series) -> str:
    euclidean_distances = np.sqrt(((dataframe.drop(target_col, axis=1) - y) ** 2).sum(axis=1))
    nearest_neighbors = euclidean_distances.nsmallest(k).index
    return (dataframe
    .loc[nearest_neighbors, target_col]
    .mode()
    .loc[0])


def distance_weighted_knn(dataframe: pd.DataFrame, target_col: str, k: int, y: pd.Series) -> str:
    euclidean_distances = np.sqrt(((dataframe.drop(target_col, axis=1) - y) ** 2).sum(axis=1))
    nearest_neighbors = euclidean_distances.nsmallest(k).index
    target_values = dataframe.loc[nearest_neighbors, target_col]
    weights = 1 / ((euclidean_distances[nearest_neighbors] + 1e-10) ** 2)
    weights /= max(weights)
    unique_target_values = target_values.unique()

    weighted_sums = {}
    for value in unique_target_values:
        indices = target_values[target_values == value].index
        weighted_sum = np.sum(weights[indices])
        weighted_sums[value] = weighted_sum

    return max(weighted_sums, key=weighted_sums.get)


def attribute_weighted_knn(dataframe: pd.DataFrame, target_col: str, k: int, y: str) -> str:
    ig_cols = {col: information_gain_col(dataframe, target_col, col)
               for col in dataframe.columns
               if col != target_col}
    max_value = max(ig_cols.values())

    ig_cols_scaled = {key: value / max_value
                      for key, value in ig_cols.items()}

    weighted_data = (dataframe
                     .drop(target_col, axis=1)
                     .apply(lambda col: ig_cols_scaled[col.name] * col))

    euclidian_distances = np.sqrt(((weighted_data - y) ** 2).sum(axis=1))
    nearest_neighbors = euclidian_distances.nsmallest(k).index
    return (dataframe
    .loc[nearest_neighbors, target_col]
    .mode()
    .loc[0])


def combined_weighted_knn(dataframe: pd.DataFrame, target_col: str, k: int, y: pd.Series) -> str:
    ig_cols = {col: information_gain_col(dataframe, target_col, col)
               for col in dataframe.columns
               if col != target_col}
    max_value = max(ig_cols.values())

    ig_cols_scaled = {key: value / max_value
                      for key, value in ig_cols.items()}

    weighted_data = dataframe.drop(target_col, axis=1).apply(lambda col: ig_cols_scaled[col.name] * col)
    euclidean_distances = np.sqrt(((weighted_data - y) ** 2).sum(axis=1))
    inverse_distances = 1 / (euclidean_distances + 1e-10) ** 2
    inverse_distances /= max(inverse_distances)

    weighted_data = weighted_data * inverse_distances.values[:, None]
    euclidean_distances = np.sqrt(((weighted_data - y) ** 2).sum(axis=1))
    nearest_neighbors = euclidean_distances.nsmallest(k).index
    return (dataframe
    .loc[nearest_neighbors, target_col]
    .mode()
    .loc[0])