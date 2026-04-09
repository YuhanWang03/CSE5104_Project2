PARAM_GRID = {
    'hidden_sizes': [(50,), (100,), (200,), (50, 50), (100, 50)],
    'activation':   ['relu', 'tanh'],
    'lr':           [0.0001, 0.001, 0.01],
    'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'batch_size':   [32, 64, 128, 256],
}
SEARCH_TYPE     = 'grid'   # 600 combinations
N_ITER          = 600
CV_FOLDS        = 5
SCORING         = 'balanced_accuracy'
PLOT_TYPE       = 'bar'
PLOT_TITLE      = 'ANN Cross-Validation'
PLOT_X_AXIS     = None
CV_RESULTS_FILE = 'ANN_cv_results.csv'
