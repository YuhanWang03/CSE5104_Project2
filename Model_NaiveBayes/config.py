PARAM_GRID = {
    'var_smoothing': [1e-9, 1e-5, 1e-3, 1e-2, 1e-1],
}
SEARCH_TYPE     = 'grid'   # 5 combinations
N_ITER          = 5
CV_FOLDS        = 5
SCORING         = 'balanced_accuracy'
PLOT_TYPE       = 'line'
PLOT_TITLE      = 'Naive Bayes Cross-Validation'
PLOT_X_AXIS     = 'param_var_smoothing'
CV_RESULTS_FILE = 'NB_cv_results.csv'
