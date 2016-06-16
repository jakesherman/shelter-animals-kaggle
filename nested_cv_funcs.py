def nested_cv(X, y, estimator, params, scoring = 'log_loss', cv = 5, 
              n_jobs = -1, verbose = True):
    """Performs 5-fold nested cross-validation ([cv] folds in each loop) on an 
    estimator given a parameter grid of hyperparamaters to optimize over using 
    grid search.
    """
    start_time = time()
    inner_loop = GridSearchCV(estimator, params, cv = cv, n_jobs = n_jobs, 
    scoring = scoring)
    score = np.absolute(np.mean(cross_val_score(inner_loop, X, y, cv = cv, 
        n_jobs = 1)))
    if verbose:
        time_elapsed = time() - start_time
        print 'Model:', estimator
        print 'Score:', score
        print 'Time elapsed:', round(time_elapsed / 60, 1), '\n'
    return score


def model_selection(X, y, estimators_params, scoring = 'log_loss', cv = 5, 
                    n_jobs = -1, refit = True, higher_is_better = False, 
                    verbose = True):
    """Evalute multiple estimators using nested cross-validation. If refit is 
    True, the best scoring estimator is returned as part of a [cv]-fold 
    GridSearchCV estimator,such that fitting that model with X, y will find the 
    optimal hyperparameters and return a final model that can be used to make 
    predictions. [estimators_parms] is a dictionary where the key is the 
    estimator and the value is the hyperparameter grid for that estimator.
    """
    scores = []
    for estimator, params in estimators_params.items():
        score = nested_cv(X, y, estimator, params, scoring, cv, n_jobs, verbose)
        scores.append([score, estimator])
    scores = sorted(scores, reverse = higher_is_better)
    best_model = scores[0][1]
    if refit:
        return GridSearchCV(best_model, estimators_params[best_model], 
                            cv = cv, n_jobs = n_jobs, scoring = scoring)
    else:
        return best_model