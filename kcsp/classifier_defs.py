# Dictionary with specification of classifiers used from scikit-learn

CLASSIFIERS = {
    '1': {
        'name': 'LinearSVC',
        'params': {
            'Scaler': {'name': 'l2'},
            'C': 1,
            'dual': True,
            'fit_intercept': True,
            'random_state': 13,  # can't use, yet, a numpy generator
            'class_weight': 'balanced'
        }
    },
    '2': {'name': 'MultinomialNB'},
    '3': {
        'name': 'LinearSVC',
        'params': {
            'Scaler': {'name': 'tf-idf'},
            'C': 1,
            'dual': True,
            'fit_intercept': True,
            'random_state': 13,
            'class_weight': 'balanced'
        }
    },    
    '4': {
        'name': 'LogisticRegression',
        'params': {
            'Scaler': {'name': 'l2'},
            'C': 1,
            'dual': True,
            'fit_intercept': True,
            'intercept_scaling': 1,  # change it?
            'random_state': 13,
            'class_weight': 'balanced',
            'solver': 'liblinear'
        }
    },
    '5': {
        'name': 'LogisticRegression',
        'params': {
            'Scaler': {'name': 'tf-idf'},
            'C': 1,
            'dual': True,
            'fit_intercept': True,
            'intercept_scaling': 1,  # change it?
            'random_state': 13,
            'class_weight': 'balanced',
            'solver': 'liblinear'
        }
    },
    '6': {
        'name': 'LinearSVC',
        'params': {
            'Scaler': {'name': 'min-max'},
            'C': 1,
            'dual': True,
            'fit_intercept': False,
            'random_state': 13,
            'class_weight': 'balanced'
        }
    },
    '7': {
        'name': 'LinearSVC',
        'params': {
            'Scaler': {'name': 'StandardScaler'},
            'C': 1,
            'dual': True,
            'fit_intercept': False,
            'random_state': 13,
            'class_weight': 'balanced'
        }
    },
    '8': {
        'name': 'LinearSVC',
        'params': {
            'Scaler': {'name': 'RobustScaler'},
            'C': 1,
            'dual': True,
            'fit_intercept': False,
            'random_state': 13,
            'class_weight': 'balanced'
        }
    },
    '9': {
        'name': 'LinearSVC',
        'params': {
            'Scaler': {'name': 'tf-idf'},
            'loss': 'hinge',
            'C': 1,
            'dual': True,
            'fit_intercept': True,
            'random_state': 13,
            'class_weight': 'balanced'
        }
    }}
