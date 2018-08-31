import csv

from math import sqrt
from sklearn import svm, tree
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from random import shuffle

'''
Dados reais - 107 flows
    normal = 100
    anomalos = 7

Dados artificiais - 41 flows
    normal = 32
    anomalos = 9
'''

training_set = []
training_set_classification = []

evaluation_set = []

voting_predictions = []
stacking_predictions = []
adaboost_predictions = []
bagging_predictions = []

# Classificadores base
classifier_knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute', p=2)
classifier_svm = svm.SVC(kernel='linear')
classifier_dt = tree.DecisionTreeClassifier(criterion='entropy')

svm_predictions = []
knn_predictions = []
dt_predictions = []

anom_flows_count = 0
normal_flows_count = 0

voting_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
stacking_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
bagging_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
adaboost_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
svm_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
knn_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
dt_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
    'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}


def not_main():
    experiments_times = 50

    selectTrainingSet(0.3)
    selectEvaluationSet(0.7)

    classifier_knn.fit(training_set, training_set_classification)
    classifier_svm.fit(training_set, training_set_classification)
    classifier_dt.fit(training_set, training_set_classification)

    print('>>> Dados reais')
    print('Tamanho do conjunto de avaliação: ', len(evaluation_set))

    runNTimes(experiments_times)

    print('')
    print('')
    print('>>> Dados Anderson ')
    resetVariables()

    dadosAndersonSelectTrainingSet(0.3)
    dadosAndersonSelectEvaluationSet(0.7)

    print('Tamanho do conjunto de avaliação: ', len(evaluation_set))

    classifier_knn.fit(training_set, training_set_classification)
    classifier_svm.fit(training_set, training_set_classification)
    classifier_dt.fit(training_set, training_set_classification)

    runNTimes(experiments_times)


def main():
    crossValidation()


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


def crossValidation():
    # k = 5

    # dividir o conjunto de dados reais em k subconjuntos de tamanho aproximadamente igual
    real_normal_instances = getRealNormalInstances() # 100 instancias
    real_anom_instances = getRealAnomalousInstances() # 7 instancias

    # dividir o conjunto de dados artificiais em k subconjuntos de tamanho aproximadamente igual
    # normal = 32, anom = 9
    artificial_normal_instances, artificial_anom_instances = getArtificialInstances()

    #crossValidate()


def get5Folds(normal_instances, anom_instances):
    shuffle(normal_instances)

    fold1 = []
    fold2 = []
    fold3 = []
    fold4 = []
    fold5 = []

    for instance in normal_instances:
        pass



def crossValidate(folds, k):
    for i in range(0, k):
        # usar k-1 folds para treinamento, e o fold restante, para avaliação
        # em cada repetição, um fold diferente é usado para teste
        pass



def calculateMeanStatistics(times):
    technique_name = ['Voting', 'AdaBoost', 'Bagging', 'Stacking', 'SVM', 'KNN', 'DT']
    j = 0

    for statistics in [voting_statistics, adaboost_statistics, bagging_statistics, stacking_statistics, svm_statistics, knn_statistics, dt_statistics]:
        false_negatives_count = 0
        false_positives_count = 0
        true_negatives_count = 0
        true_positives_count = 0
        recall_count = 0
        precision_count = 0
        accuracy_count = 0
        error_rate_count = 0

        variance = 0
        std_deviation = 0


        for i in range(0, times):
            false_negatives_count = false_negatives_count + statistics['false_negatives'][i]
            false_positives_count = false_positives_count + statistics['false_positives'][i]
            true_negatives_count = true_negatives_count + statistics['true_negatives'][i]
            true_positives_count = true_positives_count + statistics['true_positives'][i]
            recall_count = recall_count + statistics['recall'][i]
            precision_count = precision_count + statistics['precision'][i]
            accuracy_count = accuracy_count + statistics['accuracy'][i]
            error_rate_count = error_rate_count + statistics['error_rate'][i]

        # Média dos valores
        false_negatives_mean = false_negatives_count/times
        false_positives_mean = false_positives_count/times
        true_negatives_mean = true_negatives_count/times
        true_positives_mean = true_positives_count/times
        recall_mean = recall_count/times
        precision_mean = precision_count/times
        accuracy_mean = accuracy_count/times
        error_rate_mean = error_rate_count/times

        # Cálculo da variância
        false_negatives_variance = calculateVariance(statistics['false_negatives'], false_negatives_mean)
        false_positives_variance = calculateVariance(statistics['false_positives'], false_positives_mean)
        true_negatives_variance = calculateVariance(statistics['true_negatives'], true_negatives_mean)
        true_positives_variance = calculateVariance(statistics['true_positives'], true_positives_mean)
        recall_variance = calculateVariance(statistics['recall'], recall_mean)
        precision_variance = calculateVariance(statistics['precision'], precision_mean)
        accuracy_variance = calculateVariance(statistics['accuracy'], accuracy_mean)
        error_rate_variance = calculateVariance(statistics['error_rate'], error_rate_mean)

        # Cálculo do desvio padrão
        false_negatives_std_deviation = sqrt(false_negatives_variance)
        false_positives_std_deviation = sqrt(false_positives_variance)
        true_positives_std_deviation = sqrt(true_positives_variance)
        true_negatives_std_deviation = sqrt(true_negatives_variance)
        recall_std_deviation = sqrt(recall_variance)
        precision_std_deviation = sqrt(precision_variance)
        accuracy_std_deviation = sqrt(accuracy_variance)
        error_rate_std_deviation = sqrt(error_rate_variance)


        print('>> {0}'.format(technique_name[j]))
        print('Falsos negativos: {0} ; var = {1} ; dp = {2}'.format(false_negatives_mean, false_negatives_variance, false_negatives_std_deviation))
        print('Falsos positivos: {0} ; var = {1} ; dp = {2}'.format(false_positives_mean, false_positives_variance, false_positives_std_deviation))
        print('Verdadeiros negativos: {0} ; var = {1} ; dp = {2}'.format(true_negatives_mean, true_negatives_variance, true_negatives_std_deviation))
        print('Verdadeiros positivos: {0} ; var = {1} ; dp = {2}'.format(true_positives_mean, true_positives_variance, true_positives_std_deviation))
        print('Recall: {0} ; var = {1} ; dp = {2}'.format(recall_mean, recall_variance, recall_std_deviation))
        print('Precisão: {0} ; var = {1} ; dp = {2}'.format(precision_mean, precision_variance, precision_std_deviation))
        print('Acurácia: {0} ; var = {1} ; dp = {2}'.format(accuracy_mean, accuracy_variance, accuracy_std_deviation))
        print('Taxa de erro: {0}; var = {1}; dp = {2}'.format(error_rate_mean, error_rate_variance, error_rate_std_deviation))
        j = j + 1


def calculateVariance(list_values, mean):
    diff_sum = 0

    for i in range(0, len(list_values)):
        diff_sum = diff_sum + pow((list_values[i] - mean), 2)

    variance = diff_sum/len(list_values)
    return variance


def runNTimes(times):
    for i in range(0, times):
        voting(evaluation_set)
        adaboost(evaluation_set)
        bagging(evaluation_set)
        stacking(evaluation_set)

        svm(evaluation_set)
        knn(evaluation_set)
        decisionTree(evaluation_set)

    calculateMeanStatistics(times)


def resetVariables():
    global training_set
    training_set = []
    global training_set_classification
    training_set_classification = []

    global evaluation_set
    evaluation_set = []

    global voting_predictions
    voting_predictions = []
    global stacking_predictions
    stacking_predictions = []
    global adaboost_predictions
    adaboost_predictions = []
    global bagging_predictions
    bagging_predictions = []

    global svm_predictions
    svm_predictions = []
    global knn_predictions
    knn_predictions = []
    global dt_predictions
    dt_predictions = []

    global anom_flows_count
    anom_flows_count = 0
    global normal_flows_count
    normal_flows_count = 0

    global voting_statistics, stacking_statistics, bagging_statistics, adaboost_statistics, svm_statistics, knn_statistics, dt_statistics
    voting_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
    stacking_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
    bagging_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
    adaboost_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
    svm_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
    knn_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}
    dt_statistics = {'false_negatives': [], 'false_positives': [], 'true_negatives': [],'true_positives': [],
        'recall': [], 'precision': [], 'accuracy': [], 'error_rate': []}



def calculateStatistics(predictions):
    # Calcula a acurácia do método
    false_positives = 0
    false_negatives = 0

    total_predictions =  len(voting_predictions)

    # Utilizado para o cálculo de precisão
    classified_as_anom_count = 0

    wright_normal_count = 0
    global normal_flows_count
    for i in range(0, normal_flows_count):
        if predictions[i] == 0:
            wright_normal_count = wright_normal_count + 1
        else:
            # fluxo normal que foi considerado anômalo
            false_positives = false_positives + 1
            classified_as_anom_count = classified_as_anom_count + 1

    wright_anom_count = 0
    # global normal_flows_count, anom_flows_count
    for i in range(normal_flows_count, normal_flows_count + anom_flows_count):
        if predictions[i] == 1:
            wright_anom_count = wright_anom_count + 1
            classified_as_anom_count = classified_as_anom_count + 1
        else:
            # Fluxo anômalo que foi considerado normal
            false_negatives = false_negatives + 1

    wright_count = wright_normal_count + wright_anom_count
    accuracy = wright_count / total_predictions
    #print('> {0}//{1}'.format(wright_count, len(voting_predictions)))


    # Recall é porcentagem de positivos que conseguimos acertar
    # 5 está hardcoded -> numero de fluxos anomalos no evaluation set
    # global anom_flows_count
    recall = wright_anom_count / anom_flows_count

    # Precision é a porcentagem das previsões positivas que está correta
    # Número de flows anômalos que classificamos como anômalos
    if classified_as_anom_count > 0:
        precision = wright_anom_count / classified_as_anom_count
    else:
        # eh isso mesmo?
        precision = 0


    # Calcula a taxa de erro (error rate)
    error_rate = (false_negatives + false_positives) / total_predictions

    true_negatives = wright_normal_count
    true_positives = wright_anom_count
    # Retorna uma lista com todas as estatísticas
    return [false_negatives, false_positives, true_negatives, true_positives, recall, precision, accuracy, error_rate]


def voting(evaluation_set):
    meta_learner = VotingClassifier(estimators=[
        ('knn', classifier_knn),
        ('svm', classifier_svm),
        ('dt', classifier_dt)],
        voting='hard')
    meta_learner = meta_learner.fit(training_set, training_set_classification)

    result = meta_learner.predict(evaluation_set)

    global voting_predictions
    voting_predictions = []
    voting_predictions = list(result.tolist())

    global voting_statistics
    statistics = calculateStatistics(voting_predictions)
    voting_statistics['false_negatives'].append(statistics[0])
    voting_statistics['false_positives'].append(statistics[1])
    voting_statistics['true_negatives'].append(statistics[2])
    voting_statistics['true_positives'].append(statistics[3])
    voting_statistics['recall'].append(statistics[4])
    voting_statistics['precision'].append(statistics[5])
    voting_statistics['accuracy'].append(statistics[6])
    voting_statistics['error_rate'].append(statistics[7])

    # pegar dados com base no que eu to executando - anderson ou reais
    # classificacao_real = []
    # rocCurve(result, )


def adaboost(evaluation_test):
    meta_learner = AdaBoostClassifier(base_estimator=classifier_dt, n_estimators=15, algorithm='SAMME')
    meta_learner = meta_learner.fit(training_set, training_set_classification)

    result = meta_learner.predict(evaluation_test)
    global adaboost_predictions
    adaboost_predictions = []
    adaboost_predictions = list(result.tolist())

    global adaboost_statistics
    statistics = calculateStatistics(adaboost_predictions)
    adaboost_statistics['false_negatives'].append(statistics[0])
    adaboost_statistics['false_positives'].append(statistics[1])
    adaboost_statistics['true_negatives'].append(statistics[2])
    adaboost_statistics['true_positives'].append(statistics[3])
    adaboost_statistics['recall'].append(statistics[4])
    adaboost_statistics['precision'].append(statistics[5])
    adaboost_statistics['accuracy'].append(statistics[6])
    adaboost_statistics['error_rate'].append(statistics[7])


def bagging(evaluation_test):
    meta_learner = BaggingClassifier(base_estimator=classifier_dt, n_estimators=10)
    meta_learner = meta_learner.fit(training_set, training_set_classification)

    result = meta_learner.predict(evaluation_test)
    global bagging_predictions
    bagging_predictions = []
    bagging_predictions = list(result.tolist())

    global bagging_statistics
    statistics = calculateStatistics(bagging_predictions)
    bagging_statistics['false_negatives'].append(statistics[0])
    bagging_statistics['false_positives'].append(statistics[1])
    bagging_statistics['true_negatives'].append(statistics[2])
    bagging_statistics['true_positives'].append(statistics[3])
    bagging_statistics['recall'].append(statistics[4])
    bagging_statistics['precision'].append(statistics[5])
    bagging_statistics['accuracy'].append(statistics[6])
    bagging_statistics['error_rate'].append(statistics[7])



def stacking(evaluation_set):
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

    # TODO: As predições do nível 1 não deveriam ser appendadas ao conjunto de features?

    ## Nível 2 stacking - Treinamento
    meta_learner_dt = tree.DecisionTreeClassifier()
    meta_learner_dt.fit(training_set, predictions_level_1)

    result = meta_learner_dt.predict(evaluation_set)
    global stacking_predictions
    stacking_predictions = []
    stacking_predictions = list(result.tolist())

    global stacking_statistics
    statistics = calculateStatistics(stacking_predictions)
    stacking_statistics['false_negatives'].append(statistics[0])
    stacking_statistics['false_positives'].append(statistics[1])
    stacking_statistics['true_negatives'].append(statistics[2])
    stacking_statistics['true_positives'].append(statistics[3])
    stacking_statistics['recall'].append(statistics[4])
    stacking_statistics['precision'].append(statistics[5])
    stacking_statistics['accuracy'].append(statistics[6])
    stacking_statistics['error_rate'].append(statistics[7])



def svm(evaluation_set):
    global classifier_svm, svm_predictions
    result = classifier_svm.predict(evaluation_set)

    svm_predictions = []
    svm_predictions = list(result.tolist())

    global svm_statistics
    statistics = calculateStatistics(svm_predictions)
    svm_statistics['false_negatives'].append(statistics[0])
    svm_statistics['false_positives'].append(statistics[1])
    svm_statistics['true_negatives'].append(statistics[2])
    svm_statistics['true_positives'].append(statistics[3])
    svm_statistics['recall'].append(statistics[4])
    svm_statistics['precision'].append(statistics[5])
    svm_statistics['accuracy'].append(statistics[6])
    svm_statistics['error_rate'].append(statistics[7])


def knn(evaluation_set):
    global classifier_knn, knn_predictions
    result = classifier_knn.predict(evaluation_set)

    knn_predictions = []
    knn_predictions = list(result.tolist())

    global knn_statistics
    statistics = calculateStatistics(knn_predictions)
    knn_statistics['false_negatives'].append(statistics[0])
    knn_statistics['false_positives'].append(statistics[1])
    knn_statistics['true_negatives'].append(statistics[2])
    knn_statistics['true_positives'].append(statistics[3])
    knn_statistics['recall'].append(statistics[4])
    knn_statistics['precision'].append(statistics[5])
    knn_statistics['accuracy'].append(statistics[6])
    knn_statistics['error_rate'].append(statistics[7])


def decisionTree(evaluation_set):
    global classifier_dt, dt_predictions
    result = classifier_dt.predict(evaluation_set)

    dt_predictions = []
    dt_predictions = list(result.tolist())

    global dt_statistics
    statistics = calculateStatistics(dt_predictions)
    dt_statistics['false_negatives'].append(statistics[0])
    dt_statistics['false_positives'].append(statistics[1])
    dt_statistics['true_negatives'].append(statistics[2])
    dt_statistics['true_positives'].append(statistics[3])
    dt_statistics['recall'].append(statistics[4])
    dt_statistics['precision'].append(statistics[5])
    dt_statistics['accuracy'].append(statistics[6])
    dt_statistics['error_rate'].append(statistics[7])



if __name__ == '__main__':
    main()
