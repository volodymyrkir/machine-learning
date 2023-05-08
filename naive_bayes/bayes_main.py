from bayes_algorithm import NaiveBayesManager

from utils import get_data

from utils import IRIS_COLS, IRIS_NAME, IRIS_TARGET
from utils import CONGRESS_COLS, CONGRESS_TARGET, CONGRESS_NAME
from utils import WINE_TARGET, WINE_COLS, WINE_NAME
from utils import BREAST_TARGET, BREAST_NAME

if __name__ == '__main__':
    iris_data = get_data('iris.data', IRIS_COLS)

    congress_data = get_data('congress.data', CONGRESS_COLS)

    wine_data = get_data('wine.data', WINE_COLS)

    breast_data = get_data('breast.csv')

    bayes_manager = NaiveBayesManager(list(range(50, 91, 10)))

    bayes_manager.import_dataset(iris_data, IRIS_NAME, IRIS_TARGET, True)
    bayes_manager.import_dataset(wine_data, WINE_NAME, WINE_TARGET, True)
    bayes_manager.import_dataset(congress_data, CONGRESS_NAME, CONGRESS_TARGET, False, True, '?')
    bayes_manager.import_dataset(breast_data, BREAST_NAME, BREAST_TARGET, True)
    results = bayes_manager.collect_results()
