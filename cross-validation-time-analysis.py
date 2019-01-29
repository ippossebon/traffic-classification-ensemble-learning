import csv
import time

from math import sqrt
from random import shuffle
from sklearn import svm, tree
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


'''
Dados reais - 107 flows
    normal = 100
    anomalos = 7

Dados artificiais - 41 flows
    normal = 32
    anomalos = 9
'''

# Classificadores base
classifier_knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute', p=2)
classifier_svm = svm.SVC(kernel='linear')
classifier_dt = tree.DecisionTreeClassifier(criterion='entropy')


def main():
    real_normal_instances = getRealNormalInstances() # 100 instancias
    real_anom_instances = getRealAnomalousInstances() # 7 instancias

    # normal = 32, anom = 9
    artificial_normal_instances, artificial_anom_instances = getArtificialInstances()

    weights = [0.5]

    print('* Real instances *')
    repeatedCrossValidation(real_normal_instances, real_anom_instances, 5, 5)


    print('* Artificial instances *')
    repeatedCrossValidation(artificial_normal_instances, artificial_anom_instances, 5, 5)



def getRealNormalInstances():
    normal_instances = []

    with open('csv-files-features/feats_normal100.csv', 'r') as csv_file:
        content = csv.reader(csv_file, delimiter=',')

        for line in content:
            inst = [float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12])]
            normal_instances.append(inst)
    return normal_instances


def getRealAnomalousInstances():
    anom_instances = []

    with open('csv-files-features/anom-flows/booters-feats.csv', 'r') as csv_file:
        content = csv.reader(csv_file, delimiter=',')

        for line in content:
            inst = [float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12])]
            anom_instances.append(inst)

    return anom_instances


def getArtificialInstances():
    num_instances = 0
    normal_instances = []
    anom_instances = []

    with open('csv-files-features/feats_anderson.csv', 'r') as csv_file:
        content = csv.reader(csv_file, delimiter=',')

        for line in content:
            inst = [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])]

            if num_instances < 9:
                # Primeiras 9 instâncias são anômalas
                anom_instances.append(inst)
            else:
                normal_instances.append(inst)

            num_instances = num_instances + 1

    return normal_instances, anom_instances


def repeatedCrossValidation(normal_set, anom_set, k, repetitions):
    k_fold = KFold(n_splits=k)

    # Repete o processo K = 5 vezes
    original_set = normal_set + anom_set

    voting_statistics = []
    adaboost_statistics = []
    bagging_statistics = []
    stacking_statistics = []
    svm_statistics = []
    knn_statistics = []
    dt_statistics = []

    voting_time = []
    adaboost_time = []
    bagging_time = []
    stacking_time = []
    svm_time = []
    knn_time = []
    dt_time = []

    technique_name = ['- Voting -', '- AdaBoost -', '- Bagging -', '- Stacking -', '- SVM -', '- KNN -', '- DT -']
    j = 0

    for i in range(repetitions):
        shuffle(original_set)

        for train_indices, test_indices in k_fold.split(original_set):
            train_set = []
            train_set_classification = []
            test_set = []

            normal_flows_in_evaluation_set = 0
            anom_flows_in_evaluation_set = 0

            # Separa o conjuntos de treinamento e a correspondente classificação
            for index in train_indices:
                train_set.append(original_set[index])

                if original_set[index] in anom_set:
                    train_set_classification.append(1)
                else:
                    train_set_classification.append(0)

            # Separa o conjunto de avaliação
            for index in test_indices:
                test_set.append(original_set[index])

                # Contabiliza número de fluxos anômalos e normais no conjunto
                # de avaliação (utilizado para estatísticas)
                if original_set[index] in anom_set:
                    anom_flows_in_evaluation_set = anom_flows_in_evaluation_set + 1
                else:
                    normal_flows_in_evaluation_set = normal_flows_in_evaluation_set + 1

            # Treinamento
            classifier_knn.fit(train_set, train_set_classification)
            classifier_svm.fit(train_set, train_set_classification)
            classifier_dt.fit(train_set, train_set_classification)

            # # Avaliação
            start_time = time.time()
            predictions = voting(train_set, train_set_classification, test_set[0:20])
            time_spent = time.time() - start_time
            voting_time.append(time_spent)

            start_time = time.time()
            predictions = adaboost(train_set, train_set_classification, test_set[0:20])
            time_spent = time.time() - start_time
            adaboost_time.append(time_spent)

            start_time = time.time()
            predictions = bagging(train_set, train_set_classification, test_set[0:20])
            time_spent = time.time() - start_time
            bagging_time.append(time_spent)

            start_time = time.time()
            predictions = stacking(train_set, train_set_classification, test_set[0:20])
            time_spent = time.time() - start_time
            stacking_time.append(time_spent)


            start_time = time.time()
            predictions = svm(test_set[0:20])
            time_spent = time.time() - start_time
            svm_time.append(time_spent)


            start_time = time.time()
            predictions = knn(test_set[0:20])
            time_spent = time.time() - start_time
            knn_time.append(time_spent)

            start_time = time.time()
            predictions = decisionTree(test_set[0:20])
            time_spent = time.time() - start_time
            dt_time.append(time_spent)

    getTimeMeasurementsPerInstance(voting_time, adaboost_time, bagging_time, stacking_time, svm_time, knn_time, dt_time)

def getTimeMeasurementsPerInstance(voting_time, adaboost_time, bagging_time, stacking_time, svm_time, knn_time, dt_time):
    technique_name = ['- Voting -', '- AdaBoost -', '- Bagging -', '- Stacking -', '- SVM -', '- KNN -', '- DT -']
    j = 0

    for statistics_time in [voting_time, adaboost_time, bagging_time, stacking_time, svm_time, knn_time, dt_time]:
        average_time_per_instance = float(sum(statistics_time)) / float(len(statistics_time))
        print('%s = %s seconds per instance', technique_name[j], average_time_per_instance)
        j = j + 1


def voting(training_set, training_set_classification, evaluation_set):
    meta_learner = VotingClassifier(estimators=[
        ('knn', classifier_knn),
        ('svm', classifier_svm),
        ('dt', classifier_dt)],
        voting='hard')
    meta_learner = meta_learner.fit(training_set, training_set_classification)

    result = meta_learner.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions


def adaboost(training_set, training_set_classification, evaluation_set):
    meta_learner = AdaBoostClassifier(
        base_estimator=classifier_dt,
        n_estimators=15,
        algorithm='SAMME'
    )
    meta_learner = meta_learner.fit(training_set, training_set_classification)

    result = meta_learner.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions

def bagging(training_set, training_set_classification, evaluation_set):
    meta_learner = BaggingClassifier(base_estimator=classifier_dt, n_estimators=10)
    meta_learner = meta_learner.fit(training_set, training_set_classification)

    result = meta_learner.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions


def stacking(training_set, training_set_classification, evaluation_set):
    result = []

    ## Nível 1 stacking já treinado na main
    predictions_knn = []
    predictions_svm = []
    predictions_dt = []

    for instance in training_set:
        predictions_knn.append(classifier_knn.predict([instance]))
        predictions_svm.append(classifier_svm.predict([instance]))
        predictions_dt.append(classifier_dt.predict([instance]))

    predictions_level_1 = []

    # Combina as predições do nível 1 com votação majoritária
    for i in range(0, len(training_set)):
        predictions_sum = predictions_knn[i] + predictions_svm[i] + predictions_dt[i]

        pred = 0
        if predictions_sum > 0:
            pred = 1

        predictions_level_1.append(pred)

    ## Nível 2 stacking - Treinamento
    meta_learner_dt = tree.DecisionTreeClassifier()
    meta_learner_dt.fit(training_set, predictions_level_1)

    result = meta_learner_dt.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions


def svm(evaluation_set):
    global classifier_svm
    result = classifier_svm.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions


def knn(evaluation_set):
    global classifier_knn
    result = classifier_knn.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions

def decisionTree(evaluation_set):
    global classifier_dt
    result = classifier_dt.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions


if __name__ == '__main__':
    main()
