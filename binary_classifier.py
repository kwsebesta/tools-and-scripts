import numpy as np
import sklearn


# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
    # linear models
    models["logistic"] = LogisticRegression()
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models["ridge-" + str(a)] = RidgeClassifier(alpha=a)
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

