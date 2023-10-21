import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def buildModule(inputShape):
    drop_out_rate = 0.5
    inputLayer = layers.Input(shape = inputShape)
    
    x = layers.Conv1D(128,3, activation = 'relu')(inputLayer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(64,3, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(32,3, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.GlobalMaxPooling1D()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)

    model = Model(inputLayer, x)
    # optimizer = tf.keras.optimizers.Adam()
    METRICS = [
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

    return model
    
def calcMes(y_true, y_pred, fixedThreshold):
    optimal_threshold = 0
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    aupr = metrics.auc(recall, precision)
    y_predTemp = np.zeros(
        (
            len(y_pred),
        )
    )
    
    


    if fixedThreshold:
        y_predTemp[np.where(y_pred >= 0.5)] = 1
    else:
        # gmeans = np.sqrt(tpr * (1-fpr))
        # ix = np.argmax(gmeans)
        # gmeansMax = gmeans[ix]
        # gmeansTR = thresholds[ix]
        # print("gmeansTR:", gmeansTR, gmeansMax, ix)
        f1_scores = (2 * precision * recall) / (precision + recall)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        y_predTemp[np.where(y_pred >= optimal_threshold)] = 1

    f1 = metrics.f1_score(y_true, y_predTemp)
    pre = metrics.precision_score(y_true, y_predTemp)
    rec = metrics.recall_score(y_true, y_predTemp)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predTemp).ravel()
    specificity = tn / (tn+fp)
    mcc = metrics.matthews_corrcoef(y_true, y_predTemp)
    acc = metrics.accuracy_score(y_true, y_predTemp)

    mesList = [
            specificity, rec, pre, acc, f1, mcc, auc, aupr
    ]

    return mesList, optimal_threshold
    
    
    
    
rootDirData = 'Data/'
rootDirSaveReper = 'Res/Draw/drugReper/'
rootDirSaveLabel = 'Res/Draw/label/'
rootDirSaveMes = 'Res/Draw/mes/'
# rootDir = ''
files = {
    'intractionMatrix':'intraction_matrix.csv',
    'virusSim':'virus_sim.csv',
    'drugSim':'drug_sim.csv',
}


intractionMatrix = pd.read_csv(rootDirData + files['intractionMatrix'], delimiter = ',', header=None, encoding='cp1252').to_numpy()
drugNames = intractionMatrix[0, 1:]
virusNames = intractionMatrix[1:, 0]
drugNames.shape, virusNames.shape
intractionMatrix = intractionMatrix[1:, 1:]


virusSim = pd.read_csv(rootDirData + files['virusSim'], delimiter = ',', header=None, encoding='cp1252').to_numpy()
virusSim = virusSim[1:, 1:]
virusSim = virusSim.astype(float)
virusSim.shape


drugSim = pd.read_csv(rootDirData + files['drugSim'], delimiter = ',', header=None, encoding='cp1252').to_numpy()
drugSim = drugSim[1:, 1:]
drugSim = drugSim.astype(float)
drugSim.shape


samples = []
for i in range(len(drugSim)):
    samples.append(np.concatenate(
            (
                virusSim[-1], drugSim[i]
            )
        )
    )
samples = np.array(samples)
samples = np.expand_dims(samples, axis = -1)
samples.shape



drugNamesSet = {}
for dn in drugNames:
    drugNamesSet[dn] = []

Y = []
X = []
Z = []
for i in range(len(virusSim)-2):
    for j in range(len(drugSim)):
        Y.append(
            intractionMatrix[i, j]
        )
        
        Z.append(
            [
                virusNames[i], drugNames[j]
            ]
        )
        
        X.append(
            np.concatenate(
                (
                    virusSim[i], drugSim[j]
                )
            )
        )

X = np.array(X).astype(float)
Y = np.array(Y).astype(float)
Z = np.array(Z)
rndIndex = np.random.choice(len(X), len(X), replace = False)
X = X[rndIndex]
Y = Y[rndIndex]
Z = Z[rndIndex]
X.shape, Y.shape, Z.shape



mesListFixed = [
    ["Set", "Fold", "Specificity", "Recall", "Precision", "Accuracy", "F1", "MCC", "AUC", "AUPR"]
]
mesListFloating = [
    ["Set", "Fold", "Specificity", "Recall", "Precision", "Accuracy", "F1", "MCC", "AUC", "AUPR", "TR"]
]

for SeT in range(10):
    rndIndex = np.random.choice(len(X), len(X), replace = False)
    X = X[rndIndex]
    Y = Y[rndIndex]
    kf = KFold(n_splits=5)
    foldCounter = 1
    for train_index, test_index in kf.split(X):
        print("Set-Fold: ", SeT, foldCounter)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        x_train = np.expand_dims(x_train, axis = -1)
        x_test = np.expand_dims(x_test, axis = -1)

        checkpoint_filepath = '/tmp/checkpoint/' +str(SeT) +'_'+ str(foldCounter)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_prc',
            mode='max',
            save_best_only=True)
        
        model = buildModule(x_train[0].shape)
        model.fit(x_train, y_train, epochs = 500, batch_size = 128
        ,validation_data=(x_test,y_test), verbose = 1, callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(monitor='val_prc', patience=100)])

        model.load_weights(checkpoint_filepath)

        y_pred = model.predict(x_test)
        y_pred = y_pred[:,0]

        np.savetxt(rootDirSaveLabel + str(SeT) + '_' + str(foldCounter) + '_gt.csv', y_test, delimiter=',')
        np.savetxt(rootDirSaveLabel + str(SeT) + '_' + str(foldCounter) + '_pred.csv', y_pred, delimiter=',')

        res =model.predict(samples)
        res = res[:,0]
        mesFixed, _ = calcMes(y_test, y_pred, True)
        mesFixed.insert(0, SeT)
        mesFixed.insert(1, foldCounter)
        mesListFixed.append(
            mesFixed
        )
        np.savetxt(rootDirSaveMes + 'fixed.csv', mesListFixed, delimiter=',', fmt='%s')
        
        mesFloating, optimalTR = calcMes(y_test, y_pred, False)
        mesFloating.insert(0, SeT)
        mesFloating.insert(1, foldCounter)
        mesFloating.append(optimalTR)
        mesListFloating.append(
            mesFloating
        )
        np.savetxt(rootDirSaveMes + 'floating.csv', mesListFloating, delimiter=',', fmt='%s')

        tempDrugList = []
        for i in range(len(res)):
            if res[i] > sum(res) / len(res):
                tempDrugList.append(
                    [
                        drugNames[i], res[i]
                    ]
                )

        # res = res[:,0]
        if len(tempDrugList) > 0:
            tempDrugList = np.array(tempDrugList)
            tempDrugList = tempDrugList[tempDrugList[:, 1].argsort()[::-1]]
            for dns in range(len(tempDrugList)):
                tempDN = tempDrugList[dns][0]
                tempDNS = [SeT, foldCounter, dns, tempDrugList[dns][1]]
    
                drugNamesSet[tempDN].append(tempDNS)
    
    
            for key in drugNamesSet.keys():
                if len(drugNamesSet[key]) > 0:
                    np.savetxt(rootDirSaveReper + key + '.csv', drugNamesSet[key], delimiter=',', fmt='%s')

        # np.savetxt(rootDirSaveReper + str(foldCounter) + '.csv', tempDrugList, delimiter=',', fmt='%s')

        foldCounter += 1
        
        
        
filenames = []
for filename in glob(rootDirSaveReper + '*.csv', recursive=True):
    filenames.append(filename) 
# filenames = glob2.glob("/*.csv")
preDrugs = []
for i in range(len(filenames)):
    preDrugs.append(filenames[i].split('\\')[1].split('.')[0])
meanScore = []
for i in range(len(filenames)):
    sumL = 0
    lenL = 0
    index = pd.read_csv(filenames[i], delimiter = ',', header = None).to_numpy()[:, :3]
    for counter in range(len(index)):
        # if index[counter][0] != 29:
        sumL += index[counter][2]
        lenL += 1
    avg = sumL / lenL
    meanScore.append(
        [
            preDrugs[i],
            avg
        ]
    )

np.savetxt(rootDirSaveMes + 'meanScoreDrug.csv', meanScore, delimiter=',', fmt='%s')