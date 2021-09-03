from copy import Error
import mne
from mne import Epochs, find_events
import numpy as np
import pickle


from analysis_tools import load_raw

sampling_rate = 250
filename = 'trained_model.sav'


clf = None

subject = 0
session = 0
ch_names = {}

def existe_clasificador():
    try:
        clf = pickle.load(open(filename, 'rb'))
        return True
    except:
        return False

def preprocess_and_train(name_csv):
    #Cargar valores de los canales en cada época
    raw = load_raw(name_csv, sfreq=sampling_rate, stim_ind=8, replace_ch_names=None, ch_ind=[0, 1, 2, 3, 4, 5])
    print(raw)

    #Guardar nombres de cada uno de los canales utilizados
    for i, chn in enumerate(raw.ch_names):
        ch_names[chn] = i


    #Aplicación de filtro de muesca de 50Hz para eliminar ruido de línea eléctrica en Europa

    raw_notch = raw.copy().notch_filter([50.0])

    #Aplicación de filtro de paso de banda de 1Hz a 17Hz
    lower = 1
    upper = 17
    raw_notch_and_filter = raw_notch.copy().filter(lower, upper, method='iir')

    #Downsampling de la señal de 250 Hz a 50 Hz (50 muestras por segundo)
    raw_notch_and_filter_downsamp = raw_notch_and_filter.copy()
    raw_notch_and_filter_downsamp.resample(50, npad="auto")

    #Uso de métodos ICA para limpiar el ruido aún más

    ica = mne.preprocessing.ICA(random_state=97)
    ica.fit(raw_notch_and_filter_downsamp)
    raw_notch_and_filter_ica = raw_notch_and_filter_downsamp.copy()
    raw_notch_and_filter_ica.load_data()
    ica.exclude = [0]
    ica.apply(raw_notch_and_filter_ica)
    

    #Obtener eventos producidos
    events = find_events(raw_notch_and_filter_downsamp, shortest_event=1)
    print(events)
 
    baseline = (0.2, 0.2)
    event_id = {'Target': 1, 'NoTarget': 2}
    reject = {'eeg': 100e-6}

    #A partir de los eventos, obtener las épocas
    epochs = Epochs(raw_notch_and_filter_downsamp, events=events, event_id=event_id, tmin=0.2, tmax=0.7, baseline=baseline, reject=reject, preload=True)
    epochs.pick_types(eeg=True)
    epochs.drop_log()

    train_testing_model(epochs)

def preprocess_and_classifier(name_csv, mark_list, box):
    
    #Cargar valores de los canales en cada época
    raw = load_raw(name_csv, sfreq=sampling_rate, stim_ind=8, replace_ch_names=None, ch_ind=[0, 1, 2, 3, 4, 5])
    print(raw)

    #Guardar nombres de cada uno de los canales utilizados
    for i, chn in enumerate(raw.ch_names):
        ch_names[chn] = i


    #Aplicación de filtro de muesca de 50Hz para eliminar ruido de línea eléctrica en Europa

    raw_notch = raw.copy().notch_filter([50.0])

    #Aplicación de filtro de paso de banda de 1Hz a 17Hz
    lower = 1
    upper = 17
    raw_notch_and_filter = raw_notch.copy().filter(lower, upper, method='iir')

    #Downsampling de la señal de 250 Hz a 50 Hz (50 muestras por segundo)
    raw_notch_and_filter_downsamp = raw_notch_and_filter.copy()
    raw_notch_and_filter_downsamp.resample(50, npad="auto")

    #Uso de métodos ICA para limpiar el ruido aún más

    ica = mne.preprocessing.ICA(random_state=97)
    ica.fit(raw_notch_and_filter_downsamp)
    raw_notch_and_filter_ica = raw_notch_and_filter_downsamp.copy()
    raw_notch_and_filter_ica.load_data()
    ica.exclude = [0]
    ica.apply(raw_notch_and_filter_ica)

    #Obtener eventos producidos
    events = find_events(raw_notch_and_filter_downsamp, shortest_event=1)


    baseline = (0.2, 0.2)
    event_id = {'Epoch': 3}
    reject = {'eeg': 100e-6}

    #A partir de los eventos, obtener las épocas
    epochs = Epochs(raw_notch_and_filter_downsamp, events=events, event_id=event_id, tmin=0.2, tmax=0.7, baseline=baseline, reject=reject, preload=True)
    epochs.pick_types(eeg=True)
    print(epochs)

    letra = ejecutar_clasificador(epochs, mark_list, box)

    return letra




from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier

from pyriemann.spatialfilters import Xdawn
import seaborn as sns
from datetime import datetime as dt


# box es la matriz gráfica que se muestra en pantalla al usuario. A partir de ella, obtenemos el caracter seleccionado
# mark_list contiene los timestamp con la fila/columna que se ha iluminado en cada momento
def ejecutar_clasificador(epochs, mark_list, box):
    global clf

    try:
        clf = pickle.load(open(filename, 'rb'))
        sc = pickle.load(open('scaler.pkl', 'rb'))
    except:
        return Error('No se puede realizar una clasificación si no se entrena el modelo')


    print("Comienzo del proceso de clasificación del experimento: ", dt.now().strftime('%d-%m-%Y, %H:%M:%S (GMT+1)'))

    X = epochs.get_data() * 1e6

    xtest_scaled = []

    for i in range(len(X)):    
        xtest_scaled.append(sc.transform(X[i]))
    
    xxtest = np.array(xtest_scaled)
        
    y_pred = clf.predict(xxtest) # testing
 
    orden = list()
    for i in mark_list:
        orden.append(mark_list[i])
    print("Orden en el que se han iluminado filas y columnas")
    print(orden)

    List = []
    for i in range(12):
        List.append(0)

    # A veces hay bad epochs que se eliminan, en el caso de que se elimine alguno, eliminamos su valor de mark_list tambien
    if (len(y_pred)!=len(mark_list)):
        i = len(mark_list)-len(y_pred)
        for aux in range(i):
            mark_list.popitem()
  
    index = 0

    # Se obtienen los P300 que han sido detectados en cada fila o columna
    for timestamp in mark_list:
        if(y_pred[index]==1):
            List[mark_list[timestamp]-1] = List[mark_list[timestamp]-1] + 1
        index = index + 1

    print("P300 detectados por filas y columna")
    print(List)

    fila = 0
    vfila = 0
    columna = 0
    vcolumna = 0

    # Calculamos que fila y columna es las mas detectada como P300
    for i in range(6):
        if List[i] > vfila:
            vfila = List[i]
            fila = i+1

    for i in range(6, 12):
        if List[i] > vcolumna:
            vcolumna = List[i]
            columna = i+1

    List.clear()
    print("FILA DETECTADA: " + str(fila))
    print("COLUMNA DETECTADA: " + str(columna))

    # Ver que fila y columna es la más elegida
    nfila = (fila - 1) * 6 + 1
    posicion = nfila + columna - 7
    
    print("Fin del proceso de clasificación del experimento: ", dt.now().strftime('%d-%m-%Y, %H:%M:%S (GMT+1)'))
    print("Letra: " + str(box[posicion].cget("text")))
    sns.despine()

    return box[posicion].cget("text")

from sklearn.metrics import classification_report
import pickle


def train_testing_model(epochs):
    global clf

    # Obtener datos y etiquetas de las epocas
    X = epochs.get_data() * 1e6

    ep = []
    et = []
    n = 0
    m = 0
    
    # Eliminar epocas NoTarget para equilibrar numero de Target y NoTarget
    for i in range(len(epochs)):
        if epochs.events[:, -1][m]==2 and n < (len(epochs['NoTarget']) - len(epochs['Target'])): # Numero de NoTargets que son eliminados
            n = n + 1  
        elif epochs.events[:, -1][m]==1:
            et.append(1)
            ep.append(X[m])
        elif epochs.events[:, -1][m]==2:
            ep.append(X[m])
            et.append(2)
        m = m + 1

    epocas = np.array(ep) # Numero de epocas Target y NoTarget equilibrado
    etiquetas = np.array(et)

    try:
        clf = pickle.load(open(filename, 'rb'))
    except:
        #clf = make_pipeline(Vectorizer(), LogisticRegression())
        #clf = make_pipeline(Vectorizer(), LDA())
        #clf = make_pipeline(Xdawn(2, classes=[1]), Vectorizer(), LDA())
        clf = make_pipeline(Vectorizer(), RandomForestClassifier())

    
    xtrain_scaled = []
    xtest_scaled = []
    sc = StandardScaler()
    preds = np.empty(len(etiquetas))
    # Instancia de validador cruzado
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Datos y Etiquetas de la clase Epochs
    data = epochs.get_data() * 1e6
    labels = epochs.events[:, -1]
    print(labels)



    #X_train, X_test, y_train, y_test = train_test_split(epochs.get_data() * 1e6, epochs.events[:, -1]==1, test_size=0.25, random_state=42)
    
    for train, test in cv.split(epocas, etiquetas):

        xtrain = epocas[train]
        xtest = epocas[test]
        ytrain = etiquetas[train]
        ytest = etiquetas[test]

        # Estandarizacion de los datos mediante StandardScaler()
        for i in range(len(xtrain)):
            sc.fit(xtrain[i])
        for i in range(len(xtrain)):    
            xtrain_scaled.append(sc.transform(xtrain[i]))
        
        for i in range(len(xtest)):
            xtest_scaled.append(sc.transform(xtest[i]))
        
        xxtrain = np.array(xtrain_scaled)
        xxtest = np.array(xtest_scaled)
        
        # Entrenamiento y evaluacion del modelo
        clf.fit(xxtrain, ytrain)
        preds[test] = clf.predict(xxtest)

        # En cada ejecución de KFold, limpiar arrays
        xtrain_scaled.clear()
        xtest_scaled.clear()

    target_names = ['Target', 'NoTarget']
    
    report = classification_report(labels, preds, target_names=target_names)
    print(report)

    scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
    
    # Comprobamos los valores obtenidos en cada ejecucion de KFold
    scores = cross_validate(estimator=clf,X=epocas,y=etiquetas,cv=cv, scoring=scoring)
    print(scores)
    
    pickle.dump(sc, open('scaler.pkl', 'wb'))
    pickle.dump(clf, open(filename,'wb'))


def preprocess_and_classifierRegion(name_csv, mark_list):
    
    #Cargar valores de los canales en cada época
    raw = load_raw(name_csv, sfreq=sampling_rate, stim_ind=8, replace_ch_names=None, ch_ind=[0, 1, 2, 3, 4, 5])
    print(raw)

    #Guardar nombres de cada uno de los canales utilizados
    for i, chn in enumerate(raw.ch_names):
        ch_names[chn] = i

    #Aplicación de filtro de muesca de 50Hz para eliminar ruido de línea eléctrica en Europa
    raw_notch = raw.copy().notch_filter([50.0])

    #Aplicación de filtro de paso de banda de 1Hz a 17Hz
    lower = 1
    upper = 17
    raw_notch_and_filter = raw_notch.copy().filter(lower, upper, method='iir')

    #Downsampling de la señal de 250 Hz a 50 Hz (50 muestras por segundo)
    raw_notch_and_filter_downsamp = raw_notch_and_filter.copy()
    raw_notch_and_filter_downsamp.resample(50, npad="auto")

    #Uso de métodos ICA para limpiar el ruido aún más

    ica = mne.preprocessing.ICA(random_state=97)
    ica.fit(raw_notch_and_filter_downsamp)
    raw_notch_and_filter_ica = raw_notch_and_filter_downsamp.copy()
    raw_notch_and_filter_ica.load_data()
    ica.exclude = [0]
    ica.apply(raw_notch_and_filter_ica)


    #Obtener eventos producidos
    events = find_events(raw_notch_and_filter_downsamp, shortest_event=1)

    
    baseline = (0.2, 0.2)
    event_id = {'Epoch': 3}
    reject = {'eeg': 100e-6}

    #A partir de los eventos, obtener las épocas
    epochs = Epochs(raw_notch_and_filter_downsamp, events=events, event_id=event_id, tmin=0.2, tmax=0.7, baseline=baseline, reject=reject, preload=True)
    epochs.pick_types(eeg=True)
    print(epochs)

    fila, columna = clasificacionRegiones(epochs, mark_list)

    return fila, columna

def clasificacionRegiones(epochs, mark_list):
    global clf

    try:
        clf = pickle.load(open(filename, 'rb'))
        sc = pickle.load(open('scaler.pkl', 'rb'))
    except:
        return Error('No se puede realizar una clasificación si no se entrena el modelo')

    print("Comienzo del proceso de clasificación del experimento: ", dt.now().strftime('%d-%m-%Y, %H:%M:%S (GMT+1)'))


    X = epochs.get_data() * 1e6
    y = epochs.events[:, -1]

    xtest_scaled = []

    # Estandarizacion de los datos para testing
    for i in range(len(X)):    
            xtest_scaled.append(sc.transform(X[i]))
    
    xxtest = np.array(xtest_scaled)
        
    y_pred = clf.predict(xxtest) # testing


    orden = list()
    for i in mark_list:
        orden.append(mark_list[i])
    print(orden)

    List = []
    for i in range(5):
        List.append(0)
    
    if (len(y_pred)!=len(mark_list)):
        i = len(mark_list)-len(y_pred)
        for aux in range(i):
            mark_list.popitem()
    
    # Se obtienen los P300 para cada una de las iluminaciones de regiones
    index = 0
    for timestamp in mark_list:
        if(y_pred[index]==1):
            List[mark_list[timestamp]-1] = List[mark_list[timestamp]-1] + 1
        index = index + 1

    print("Numero de veces que se ha detectado P300 en regiones")
    print(List)

    fila = 0
    vfila = 0
    columna = 0
    vcolumna = 0
    # Se comprueba que rregion es la mas detectada
    for i in range(2):
        if List[i] > vfila:
            vfila = List[i]
            fila = i+1
    for i in range(2, 5):
        if List[i] >= vcolumna:
            vcolumna = List[i]
            columna = i+1
    List.clear()
    print("FILA DETECTADA: " + str(fila))
    print("COLUMNA DETECTADA: " + str(columna))

    # Ver que fila y columna es la más elegida
    print("Fin del proceso de clasificación del experimento: ", dt.now().strftime('%d-%m-%Y, %H:%M:%S (GMT+1)'))
    print("Fila: " + str(fila))
    print("Columna: " + str(columna))
    sns.despine()

    return fila, columna