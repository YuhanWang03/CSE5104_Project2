PARAM_GRID = {
    'max_depth':        [3, 5, 7, 10, 12],
    'criterion':        ['gini', 'entropy'],
    'min_samples_split':[2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'max_features':     ['sqrt', 'log2'],
    'class_weight':     [None, 'balanced'],
}
SEARCH_TYPE     = 'grid'   # 1000 combinations
N_ITER          = 1000
CV_FOLDS        = 5
SCORING         = 'balanced_accuracy'
PLOT_TYPE       = 'bar'
PLOT_TITLE      = 'Decision Tree Cross-Validation'
PLOT_X_AXIS     = None
CV_RESULTS_FILE = 'DT_cv_results.csv'
