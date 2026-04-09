PARAM_GRID = {
    'n_estimators':     [50, 100, 150, 200, 300],
    'max_depth':        [5, 8, 10, 12],
    'max_features':     ['sqrt', 'log2'],
    'min_samples_split':[2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'criterion':        ['gini', 'entropy'],   # sklearn only; cuML ignores
    'class_weight':     [None, 'balanced'],    # sklearn only; cuML ignores
}
SEARCH_TYPE     = 'grid'   # 4000 combinations
N_ITER          = 4000
CV_FOLDS        = 5
SCORING         = 'balanced_accuracy'
PLOT_TYPE       = 'bar'
PLOT_TITLE      = 'Random Forest Cross-Validation'
PLOT_X_AXIS     = None
CV_RESULTS_FILE = 'RF_cv_results.csv'
