import pickle
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree

instances = [[0], [1], [2], [3]]
classification = [0, 0, 1, 1]

instances_classified = [{0: 0}, {1: 0}, {2: 1}, {3: 1}]

voting_predictions = []
stacking_predictions = []

def main():
    evaluation_set = [[5], [43], [8], [0], [1]]

    voting(evaluation_set)
    #stacking(evaluation_set)

    for e in voting_predictions:
        print(e)


# Implementação de voting: http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
def voting(evaluation_set):
    print('[voting] Treinando os modelos...')
    # KNN
    classifier_knn = KNeighborsClassifier(n_neighbors=3)
    classifier_knn.fit(instances, classification)

    # SVM
    classifier_svm = svm.SVC(kernel='linear', C = 1.0)
    classifier_svm.fit(instances, classification)

    # Decision Trees
    classifier_dt = tree.DecisionTreeClassifier()
    classifier_dt.fit(instances, classification)

    print('[voting] Utilizando os modelos para novas predicoes...')
    for instance in evaluation_set:
        #classifier_knn.predict([[1.1]])
        prediction_knn = classifier_knn.predict([instance])
        prediction_svm = classifier_svm.predict([instance])
        prediction_dt = classifier_dt.predict([instance])

        predictions_sum = prediction_knn + prediction_svm + prediction_dt

        # Votacao majoritaria
        final_prediction = 0
        if predictions_sum > 1:
             final_prediction = 1

        voting_predictions.append(final_prediction)


def stacking(evaluation_set):
    print('[stacking] Treinando os modelos...')

    ## Nível 1 stacking
    # KNN
    classifier_knn = KNeighborsClassifier(n_neighbors=3)
    classifier_knn.fit(instances, classification)

    # SVM
    classifier_svm = svm.SVC(kernel='linear', C = 1.0)
    classifier_svm.fit(instances, classification)

    # Decision Trees
    classifier_dt = tree.DecisionTreeClassifier()
    classifier_dt.fit(instances, classification)

    for instance in evaluation_set:
        #classifier_knn.predict([[1.1]])
        prediction_knn = classifier_knn.predict([instance])
        prediction_svm = classifier_svm.predict([instance])
        prediction_dt = classifier_dt.predict([instance])

    predictions_level_1 = []

    # Combina as predições do nível 1 com votação majoritária
    for i in range(evaluation_set):
        predictions_sum = prediction_knn[i] + prediction_svm[i] + prediction_dt[i]

        predictions_level_1 = 0
        if predictions_sum > 0:
            predictions_level_1 = 1


    ## Nível 2 stacking
    meta_learner_dt = tree.DecisionTreeClassifier()
    meta_learner_dt.fit(instances, predictions_level_1)

    print('[stacking] Utilizando os modelos para novas predicoes...')
    for instance in evaluation_set:
        pred = meta_learner_dt.predict([instance])
        stacking_predictions.append(pred)


# Implementação usando sklearn: https://gist.github.com/JovanSardinha/2c58bd1e7e3aa4c02affedfe7abe8a29
def bagging(evaluation_set, num_bags, num_elements):
    bags = []

    for i in range(num_bags):
        bag = generate_bag(num_elements)
        bags.append(bag)

    classifiers = []
    for bag in bags:
        classifier_dt = tree.DecisionTreeClassifier()
        classifier_dt.fit(bag.keys(), bag.values())
        classifiers.apppend(classifier_dt)

    predictions = []
    for instance in evaluation_set:
        for classifier in classifiers:
            predictions

# Bootstrap Aggregation Algorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

def generate_bag(num_elements):
    bag = []

    for i in range(num_elements):
        random_index = random.randint(0, len(instances_classified)-1)
        # element is a dict
        element[instances_classified.keys()[random_index]] = instances_classified[instances_classified.keys()[random_index]]
        bag.append(element)

    return bag

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)



def boosting(evaluation_set):
    pass



if __name__ == '__main__':
    main()
