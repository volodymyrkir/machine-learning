from ensemble_algorithm import Ensembler
from utils import get_data, replace_nulls

from utils import IRIS_COLS, IRIS_ATTRIBUTES, IRIS_NAME, IRIS_TARGET
from utils import CONGRESS_TARGET, CONGRESS_COLS, CONGRESS_NAME
from utils import WINE_TARGET, WINE_COLS, WINE_NAME, WINE_ATTRIBUTES
from k_nearest_neighbors.knn_utils import prepare_categorical, prepare_numeric
from k_nearest_neighbors.knn_utils import (simple_knn, distance_weighted_knn,
                                           attribute_weighted_knn, combined_weighted_knn)

if __name__ == '__main__':
    iris_data = get_data('iris.data', IRIS_COLS)
    iris_data_converted = prepare_numeric(iris_data, IRIS_ATTRIBUTES)

    congress_data = get_data('congress.data', CONGRESS_COLS)
    congress_data_cleaned = replace_nulls(congress_data, CONGRESS_TARGET)
    congress_data_converted = prepare_categorical(congress_data_cleaned)

    wine_data = get_data('wine.data', WINE_COLS)
    wine_data_converted = prepare_numeric(wine_data, WINE_ATTRIBUTES)

    ensembler = Ensembler()
    ensembler.import_dataset(iris_data, IRIS_NAME, IRIS_TARGET)
    ensembler.import_dataset(congress_data, CONGRESS_NAME, CONGRESS_TARGET)
    ensembler.import_dataset(wine_data, WINE_NAME, WINE_TARGET)
    ensembler.print_result_table()


