from termcolor import colored
import numpy as np
import pandas as pd
from time import time
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
from datetime import datetime

def printInfo(string):
    print(colored('[Thread] ' + string, 'yellow'))


def printError(string):
    print(colored('[Thread] ' + string, 'red'))


def printSuccess(string):
    print(colored('[Thread] ' + string, 'green'))


def record_experiment(experiment, experiment_time, csv_name):

    printSuccess("Hilo inicializado correctamente")

    dejitter = True

    '''
        Nombre del fichero con la señal capturada
    '''
    now = datetime.now()


    if csv_name == "":
        name_file = "experiments/{}/records/record_{}.csv".format(experiment, now.strftime("%d-%m-%Y"))
    else:
        name_file = csv_name

    printInfo("Buscando EEG stream...")
    print()
    streams = resolve_byprop('type', 'EEG', timeout=60) #Obtiene los streams EEG encontrados
    printInfo("streams: " + str(len(streams)))

    if len(streams) == 0:
        raise (RuntimeError("No se ha podido encontrar ningún EEG stream, prueba a reiniciar el módulo LSL"))

    printInfo("Conectando con el stream de la señal EEG...")
    for i in range(len(streams)):
        if streams[i].nominal_srate() == 250.0: #Comprobar que es el Stream correcto
            inlet_eeg = StreamInlet(streams[i]) #Construye el objeto Stream

    info = inlet_eeg.info()
    description = info.desc()

    frec_lsl = info.nominal_srate()
    chs_lsl = info.channel_count()
    printInfo("Frecuencia obtenida por LSL = {}".format(frec_lsl))
    printInfo("Número de canales obtenido por LSL = {}".format(chs_lsl))

    ch_names_default = ['O1', 'P3', 'Oz', 'Pz', 'O2', 'P4', 'X', 'X']
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    if (not ch_names or len(ch_names) < 8):
        ch_names = ch_names_default

    datos_input = []
    timestamps = []

    printInfo("Buscando Estímulo stream...")

    estimulo_stream = resolve_byprop('name', 'obci_eeg1', timeout=60)
    
    if estimulo_stream:
        printInfo("Conectando con el stream de Estimulo...")
        for i in range(len(estimulo_stream)):
            if(estimulo_stream[i].channel_count() == 1): #Se comprueba que el stream Estimulo tiene 1 canal
                index = i
        inlet_estimulo = StreamInlet(estimulo_stream[index])
    else:
        inlet_estimulo = False
        printError("No se ha podido encontrar ningún Estímulo stream")
        printInfo("Inicilizando captura sin Estímulo stream")



    estimulos = []
    inicio_captura = time()
    printInfo('Empezando a capturar experimento t=%.3f' % inicio_captura)
    while (time() - inicio_captura) < experiment_time:
        try:
            data, timestamp = inlet_eeg.pull_chunk(timeout=0.0)
            if timestamp:
                datos_input.append(data)
                timestamps.extend(timestamp)
            if inlet_estimulo:
                estimulo, timestamp = inlet_estimulo.pull_sample(timeout=0.0)
                if timestamp:
                    estimulos.append([estimulo, timestamp])
        except KeyboardInterrupt:
            printError('Cancelando captura EEG...')
            break
    time_correction = inlet_eeg.time_correction() #reasigna las marcas de tiempo al reloj local
    datos_input = np.concatenate(datos_input, axis=0)

    if chs_lsl > 8:
        datos_input = datos_input[:, :8]
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        linear_reg = LinearRegression().fit(X, y)
        timestamps = linear_reg.predict(X)
    
    datos_input = np.c_[timestamps, datos_input] #Concatena en duplas el timestamp con los datos obtenidos del EEG
    data = pd.DataFrame(data=datos_input, columns=['Timestamp'] + ch_names)

    data.insert(0, 'Time', np.arange(0, len(data) * 0.004, 0.004))
    
    if len(estimulos) != 0:
        data['Objetivo'] = 0
        for estim in estimulos:
            ix = np.argmin(np.abs(estim[1] - timestamps))
            # val = timestamps[ix]
            data.loc[ix, 'Objetivo'] = estim[0][0]
            #print(estim[0][0])
    
    data.to_csv(name_file, float_format='%.3f', index=False)

    printSuccess('Fichero ' + name_file + ' guardado correctamente.')

