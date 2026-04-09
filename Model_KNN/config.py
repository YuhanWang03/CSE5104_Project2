PARAM_GRID = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan', 'chebyshev'],
}
SEARCH_TYPE     = 'grid'   # 48 combinations
N_ITER          = 48
CV_FOLDS        = 5
SCORING         = 'balanced_accuracy'
PLOT_TYPE       = 'bar'
PLOT_TITLE      = 'KNN Cross-Validation'
PLOT_X_AXIS     = None
CV_RESULTS_FILE = 'KNN_cv_results.csv'
