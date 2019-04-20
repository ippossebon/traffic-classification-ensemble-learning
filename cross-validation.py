import csv
import time

from math import sqrt
from random import shuffle
from sklearn import svm, tree
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


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
classifier_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


def main():
    real_normal_instances = getRealNormalInstances() # 100 instancias
    real_anom_instances = getRealAnomalousInstances() # 7 instancias

    print('* Real instances *')
    start_time = time.time()
    repeatedCrossValidation(real_normal_instances, real_anom_instances, 5, 5)
    time_spent = time.time() - start_time
    print('Time spent = %s seconds', time_spent)


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
    mlp_statistics =[]

    voting_time = []
    adaboost_time = []
    bagging_time = []
    stacking_time = []
    svm_time = []
    knn_time = []
    dt_time = []

    technique_name = ['- Voting -', '- AdaBoost -', '- Bagging -', '- Stacking -', '- SVM -', '- KNN -', '- DT -', '- MLP -']
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
            classifier_mlp.fit(train_set, train_set_classification)

            # Avaliação
            # start_time = time.time()
            predictions = voting(train_set, train_set_classification, test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # voting_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            voting_statistics.append(statistics)

            start_time = time.time()
            predictions = adaboost(train_set, train_set_classification, test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # adaboost_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            adaboost_statistics.append(statistics)

            start_time = time.time()
            predictions = bagging(train_set, train_set_classification, test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # bagging_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            bagging_statistics.append(statistics)

            # start_time = time.time()
            predictions = stacking(train_set, train_set_classification, test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # stacking_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            stacking_statistics.append(statistics)

            # start_time = time.time()
            predictions = svm(test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # svm_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            svm_statistics.append(statistics)

            # start_time = time.time()
            predictions = knn(test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # knn_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            knn_statistics.append(statistics)

            start_time = time.time()
            predictions = decisionTree(test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # dt_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            dt_statistics.append(statistics)

            start_time = time.time()
            predictions = neuralNetwork(test_set)
            # time_spent = time.time() - start_time
            # avg_time_spent_per_instance = float(time_spent)/float(len(test_set))
            # mlp_time.append(avg_time_spent_per_instance)
            statistics = evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set)
            mlp_statistics.append(statistics)


    #getTimeMeasurements(voting_time, adaboost_time, bagging_time, stacking_time, svm_time, knn_time, dt_time)

    for statistics in [voting_statistics, adaboost_statistics, bagging_statistics, stacking_statistics, svm_statistics, knn_statistics, dt_statistics, mlp_statistics]:
        print(technique_name[j])
        calculateMeanStatistics(statistics, k * repetitions)
        j = j + 1


def getTimeMeasurements(voting_time, adaboost_time, bagging_time, stacking_time, svm_time, knn_time, dt_time):
    technique_name = ['- Voting -', '- AdaBoost -', '- Bagging -', '- Stacking -', '- SVM -', '- KNN -', '- DT -']
    j = 0

    for statistics_time in [voting_time, adaboost_time, bagging_time, stacking_time, svm_time, knn_time, dt_time]:
        average_time_per_instance = float(sum(statistics_time)) / float(len(statistics_time))
        print('%s = %s seconds per instance', technique_name[j], average_time_per_instance)
        j = j + 1


def evaluatePredictions(test_set, anom_set, predictions, anom_flows_in_evaluation_set):
    false_negatives_sum = 0
    false_positives_sum = 0
    true_negatives_sum = 0
    true_positives_sum = 0

    classified_as_anom_count = 0

    for i in range(len(test_set)):
        if test_set[i] in anom_set:
            # A classificação correta é de anômala
            if predictions[i] == 1:
                # Classificou como anomalo corretamente
                true_positives_sum = true_positives_sum + 1
                classified_as_anom_count = classified_as_anom_count + 1
            else:
                # Classificou como normal de forma errada
                false_negatives_sum = false_negatives_sum + 1
        else:
            # A classificação correta é de normal
            if predictions[i] == 0:
                # Classificou como normal corretamente
                true_negatives_sum = true_negatives_sum + 1
            else:
                # Classificou como anômalo de forma errada
                false_positives_sum = false_positives_sum + 1
                classified_as_anom_count = classified_as_anom_count + 1

    accuracy = (true_positives_sum + true_negatives_sum) / len(predictions)

    # Recall é porcentagem de positivos que conseguimos acertar
    if anom_flows_in_evaluation_set == 0:
        if true_positives_sum == 0:
            recall = 1
        else:
            recall = 0
    else:
        recall = true_positives_sum / anom_flows_in_evaluation_set

    # Precision é a porcentagem das previsões positivas que está correta
    # Número de flows anômalos que classificamos como anômalos
    if classified_as_anom_count > 0:
        precision = true_positives_sum / classified_as_anom_count
    else:
        precision = 0


    # Calcula a taxa de erro (error rate)
    error_rate = (false_negatives_sum + false_positives_sum) / len(predictions)

    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = (precision*recall)/(precision+recall)

    # Retorna uma lista com todas as estatísticas
    return [
        false_negatives_sum,
        false_positives_sum,
        true_negatives_sum,
        true_positives_sum,
        recall,
        precision,
        accuracy,
        error_rate,
        f1_score
    ]


# Calcula desempenho para todas as execuções
def calculateMeanStatistics(statisctics_set, k):
    false_negatives = []
    false_negatives_count = 0
    false_positives = []
    false_positives_count = 0
    true_negatives = []
    true_negatives_count = 0
    true_positives = []
    true_positives_count = 0
    recalls = []
    recall_count = 0
    precisions = []
    precision_count = 0
    accuracys = []
    accuracy_count = 0
    error_rates = []
    error_rate_count = 0
    f1_scores = []
    f1_score_count = 0

    variance = 0
    std_deviation = 0

    for statistics in statisctics_set:
        false_negatives_count = false_negatives_count + statistics[0]
        false_negatives.append(statistics[0])

        false_positives_count = false_positives_count + statistics[1]
        false_positives.append(statistics[1])

        true_negatives_count = true_negatives_count + statistics[2]
        true_negatives.append(statistics[2])

        true_positives_count = true_positives_count + statistics[3]
        true_positives.append(statistics[3])

        recall_count = recall_count + statistics[4]
        recalls.append(statistics[4])

        precision_count = precision_count + statistics[5]
        precisions.append(statistics[5])

        accuracy_count = accuracy_count + statistics[6]
        accuracys.append(statistics[6])

        error_rate_count = error_rate_count + statistics[7]
        error_rates.append(statistics[7])

        f1_score_count = f1_score_count + statistics[8]
        f1_scores.append(statistics[8])

    # Média dos valores
    false_negatives_mean = false_negatives_count/k
    false_positives_mean = false_positives_count/k
    true_negatives_mean = true_negatives_count/k
    true_positives_mean = true_positives_count/k
    recall_mean = recall_count/k
    precision_mean = precision_count/k
    accuracy_mean = accuracy_count/k
    error_rate_mean = error_rate_count/k
    f1_score_mean = f1_score_count/k


    # Cálculo da variância
    false_negatives_variance = calculateVariance(false_negatives, false_negatives_mean)
    false_positives_variance = calculateVariance(false_positives, false_positives_mean)
    true_negatives_variance = calculateVariance(true_negatives, true_negatives_mean)
    true_positives_variance = calculateVariance(true_positives, true_positives_mean)
    recall_variance = calculateVariance(recalls, recall_mean)
    precision_variance = calculateVariance(precisions, precision_mean)
    accuracy_variance = calculateVariance(accuracys, accuracy_mean)
    error_rate_variance = calculateVariance(error_rates, error_rate_mean)
    f1_score_variance = calculateVariance(f1_scores, f1_score_mean)

    # Cálculo do desvio padrão
    false_negatives_std_deviation = sqrt(false_negatives_variance)
    false_positives_std_deviation = sqrt(false_positives_variance)
    true_positives_std_deviation = sqrt(true_positives_variance)
    true_negatives_std_deviation = sqrt(true_negatives_variance)
    recall_std_deviation = sqrt(recall_variance)
    precision_std_deviation = sqrt(precision_variance)
    accuracy_std_deviation = sqrt(accuracy_variance)
    error_rate_std_deviation = sqrt(error_rate_variance)
    f1_score_std_deviation = sqrt(f1_score_variance)

    # Mostra resultados
    print('Falsos negativos: {0} ; var = {1} ; dp = {2}'.format(false_negatives_mean, false_negatives_variance, false_negatives_std_deviation))
    print('Falsos positivos: {0} ; var = {1} ; dp = {2}'.format(false_positives_mean, false_positives_variance, false_positives_std_deviation))
    print('Verdadeiros negativos: {0} ; var = {1} ; dp = {2}'.format(true_negatives_mean, true_negatives_variance, true_negatives_std_deviation))
    print('Verdadeiros positivos: {0} ; var = {1} ; dp = {2}'.format(true_positives_mean, true_positives_variance, true_positives_std_deviation))
    print('Recall: {0} ; var = {1} ; dp = {2}'.format(recall_mean, recall_variance, recall_std_deviation))
    print('Precisão: {0} ; var = {1} ; dp = {2}'.format(precision_mean, precision_variance, precision_std_deviation))
    print('Acurácia: {0} ; var = {1} ; dp = {2}'.format(accuracy_mean, accuracy_variance, accuracy_std_deviation))
    print('Taxa de erro: {0}; var = {1}; dp = {2}'.format(error_rate_mean, error_rate_variance, error_rate_std_deviation))
    print('F1-score: {0}; var = {1}; dp = {2}'.format(f1_score_mean, f1_score_variance, f1_score_std_deviation))


def calculateVariance(list_values, mean):
    diff_sum = 0

    for i in range(0, len(list_values)):
        diff_sum = diff_sum + pow((list_values[i] - mean), 2)

    variance = diff_sum/len(list_values)
    return variance



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


# multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
def neuralNetwork(evaluation_set):
    global classifier_mlp
    result = classifier_mlp.predict(evaluation_set)

    predictions = []
    predictions = list(result.tolist())

    return predictions

if __name__ == '__main__':
    main()
