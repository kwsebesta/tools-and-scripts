"""binary_classifier: script to make a framework to spot check binary
machine learning classification models
"""
import warnings
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
    # linear models
    models["logistic"] = LogisticRegression()
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for value in alpha:
        models["ridge-" + str(value)] = RidgeClassifier(alpha=value)
    models["sgd"] = SGDClassifier(max_iter=1000, tol=1e-3)
    models["pa"] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
    # non-linear models
    n_neighbors = range(1, 21)
    for k in n_neighbors:
        models["knn-" + str(k)] = KNeighborsClassifier(n_neighbors=k)
    models["cart"] = DecisionTreeClassifier()
    models["extra"] = ExtraTreeClassifier()
    models["svml"] = SVC(kernel="linear")
    models["svmp"] = SVC(kernel="poly")
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in c_values:
        models["svmr" + str(c)] = SVC(C=c)
    models["bayes"] = GaussianNB()
    # ensemble models
    n_trees = 100
    models["ada"] = AdaBoostClassifier(n_estimators=n_trees)
    models["bag"] = BaggingClassifier(n_estimators=n_trees)
    models["rf"] = RandomForestClassifier(n_estimators=n_trees)
    models["et"] = ExtraTreesClassifier(n_estimators=n_trees)
    models["gbm"] = GradientBoostingClassifier(n_estimators=n_trees)
    print("Defined %d models" % len(models))
    return models


# create a feature preparation pipeline for a model
# Below exist 4 options: none, standardize, normalize, or stand.+norm.
def pipeline_none(model):
    """No transforms pipeline"""
    return model


def pipeline_standardize(model):
    """Standardize transform pipeline"""
    steps = list()
    steps.append(("standardize", StandardScaler()))
    steps.append(("model", model))
    pipeline = Pipeline(steps=steps)
    return pipeline


def pipeline_normalize(model):
    """Normalize transform pipeline"""
    steps = list()
    steps.append(("normalize", MinMaxScaler()))
    steps.append(("model", model))
    pipeline = Pipeline(steps=steps)
    return pipeline


def pipeline_std_norm(model):
    """ Standardize and normalize pipeline"""
    steps = list()
    steps.append(("standardize", StandardScaler()))
    steps.append(("normalize", MinMaxScaler()))
    steps.append(("model", model))
    pipeline = Pipeline(steps=steps)
    return pipeline


def evaluate_model(X, y, model, folds, metric, pipe_func):
    """ Evaluate a signle model with a pipeline"""
    pipeline = pipe_func(model)
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    return scores


def robust_evaluate_model(X, y, model, folds, metric, pipe_func):
    """ Evaluate a model and try to trap errors and and hide warnings"""
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric, pipe_func)
    except:
        scores = None
    return scores


def evaluate_models(X, y, models, pipe_funcs, folds=10, metric="accuracy"):
    """ Evaluate a dict of models {name:object}, returns {name:score}"""
    results = dict()
    for name, model in models.items():
        # evaluate model under each preparation function
        for i in range(len(pipe_funcs)):
            # evaluate the model
            scores = robust_evaluate_model(X, y, model, folds, metric, pipe_funcs[i])
            # update name
            run_name = str(i) + name
            # show process
            if scores is not None:
                # store a result
                results[run_name] = scores
                mean_score, std_score = mean(scores), std(scores)
                print(">%s: %.3f (+/-%.3f)" % (run_name, mean_score, std_score))
            else:
                print(">%s: error" % run_name)
    return results


def summarize_results(results, maximize=True, top_n=10):
    """ Print and plot the top n results"""
    # check for no results
    if len(results) == 0:
        print("no results")
        return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, mean(v)) for k, v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # print the top n
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name]), std(results[name])
        print(
            "Rank=%d, Name=%s, Score=%.3f (+/- %.3f)"
            % (i + 1, name, mean_score, std_score)
        )
    # boxplot for the top n
    plt.boxplot(scores, labels=names)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.savefig("spotcheck.png")
