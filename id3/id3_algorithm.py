"""This module builds a classification tree based on id3 algorithm"""
import pandas as pd

from id3_utils import discretize_info_gain, discretize_stepwise
from utils import get_column_max_gain, split_train_test


class TreeManager:
    train_size = 0.7

    def create_tree(self, data, target_col, attributes, discrete=False):
        if discrete == 'info_gain':
            data = discretize_info_gain(data, target_col)
            print('Running gain discretization')
        elif discrete == 'stepwise':
            data = discretize_stepwise(data, 1)
            print('Running stepwise discretization')
        train_data, test_data = split_train_test(data)
        return Tree(train_data, target_col, attributes, test_data)


class Tree:
    def __init__(self,
                 train_data: pd.DataFrame,
                 target_col: str,
                 attributes: list[str],
                 test_data: pd.DataFrame):
        self._train_data = train_data
        self._test_data = test_data
        self._target_col = target_col
        self._attributes = attributes
        self._tree = self._id3(self._train_data, self._target_col, self._attributes)

    class Leaf:
        def __init__(self, data_frame: pd.DataFrame, target_col: str):
            self.predictions = data_frame[target_col].value_counts().to_dict()

    class Node:
        def __init__(self, attribute: str):
            self.attribute = attribute
            self.children = {}

        def add_child(self, value, subtree):
            self.children[value] = subtree

    def _id3(self, data_frame: pd.DataFrame, target_col: str, attributes: list[str]) -> Node | Leaf:
        target_attrs = data_frame[target_col].unique()
        if len(target_attrs) == 1 or not attributes:
            return self.Leaf(data_frame, target_col)
        best_attribute = get_column_max_gain(data_frame, target_col, attributes)
        root_node = self.Node(best_attribute)
        for value in data_frame[best_attribute].unique():
            subset = data_frame[data_frame[best_attribute] == value]
            if len(subset) == 0:
                root_node.add_child(value, self.Leaf(data_frame, target_col))
            else:
                new_attributes = [attr for attr in attributes if attr != best_attribute]
                root_node.add_child(value, self._id3(subset, target_col, new_attributes))

        return root_node

    def predict(self) -> float and dict:
        correct_predictions = 0
        predictions = []
        for i in range(len(self._test_data)):
            curr_node = self._tree
            while isinstance(curr_node, self.Node):
                value = self._test_data[curr_node.attribute].iloc[i]
                if value not in curr_node.children:
                    break
                curr_node = curr_node.children[value]
            if isinstance(curr_node, self.Leaf):
                predictions.append(curr_node.predictions)
                if self._test_data[self._target_col].iloc[i] == max(curr_node.predictions,
                                                                    key=curr_node.predictions.get):
                    correct_predictions += 1
            else:
                predictions.append({})
        accuracy = correct_predictions / len(self._test_data)
        print(f'Accuracy of tree run by is : {accuracy} \nPredictions dict - {predictions}')
        return accuracy, predictions

