PARAM_GRID = {
    'kernel':       ['poly', 'rbf', 'sigmoid'],
    'C':            [0.01, 0.1, 1, 10],
    'gamma':        ['scale', 'auto', 0.1, 1.0],
    'class_weight': [None, 'balanced'],
}
SEARCH_TYPE     = 'grid'   # 96 combinations
N_ITER          = 160
CV_FOLDS        = 5
SCORING         = 'balanced_accuracy'
PLOT_TYPE       = 'bar'
PLOT_TITLE      = 'SVM Cross-Validation'
PLOT_X_AXIS     = None
CV_RESULTS_FILE = 'SVM_cv_results.csv'
