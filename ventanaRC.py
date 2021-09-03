import time as tm
import os
import sys
from datetime import datetime
from threading import Thread
from tkinter import *
from tkinter import messagebox
from random import shuffle
import tkinter
from pylsl import StreamInfo, StreamOutlet, local_clock
from record import record_experiment #Importa función para obtener vía LSL los datos del BCI
from preprocess_classification import preprocess_and_classifier, preprocess_and_train, existe_clasificador


start_simulation = False
seriesxchar = 4
currSerie = 0
flash_time = 150
stop_time = 800
trainMode = False
testMode = False
index = -1  # Indica numero de filas/columnas que han sido iluminadas dentro de una serie
experiment_name = ""
train_text = ""
csv_name = ""
csv_name_actual = ""
mark_list = dict()
row_actual = None
col_actual = None
nletras = 0     # Para llevar el nº de letra que se está registrando en el modo libre de P300 Speller
time = 0    # Tiempo de duración de captura de datos del casco
palabra = ""
process = None

#Tiempo de la prueba
start_time_f = 0
start_time_ts = 0
stop_time_f = 0
stop_time_ts = 0

#Variables de LSL
info = None
outlet = None

# Arrays
List = []   # Lista para saber el orden en el que se iluminan filas y columnas
targets = []    #Contiene 12 posiciones (1 por cada fila/columna). 1 = Target, 0 = NoTarget
for i in range(1, 13):
    List.append(i)
    targets.append(0)
shuffle(List)

# Cambio a modo de entrenamiento de modelo
def do_training():
    global trainMode, testMode, start_simulation

    if (start_simulation == True):
        messagebox.showwarning("Alert", "You Cannot switch modes currently!")
        return

    trainMode = True
    testMode = False
    pred_label1.configure(text="Training Text")
    pred_Text1.configure(state="normal")
    pred_Text2.configure(state="normal")
    pred_label2.configure(text="Training Letter")
    pred_label3.configure(text="")

# Cambio a modo de evaluacion en tiempo real
def do_testing():
    global trainMode, testMode, start_simulation

    if (start_simulation == True):
        messagebox.showwarning("Alert", "You Cannot switch modes currently!")
        return
    
    trainMode = False
    testMode = True
    pred_label1.configure(text="Predicted Text")
    pred_Text1.configure(state="normal")
    pred_Text2.delete('1.0', END)
    pred_Text2.configure(state="disabled")
    pred_label2.configure(text="Predicted Letter")
    pred_label3.configure(text="")


# current_serie : serie por la que se encuentra una letra en concreto (4 series)
def change_color(current_serie):
    global start_simulation, index, seriesxchar, train_text, current_index, csv_name_actual, nletras
    global trainMode, currSerie, mark_list, row_actual, col_actual, csv_name, time, palabra, process

    if (start_simulation == False):
        return
    
    index = (index + 1)
    if (index >= 12):
        current_serie += 1
        index = 0
        shuffle(List)
        if (current_serie >= seriesxchar):
            if (trainMode is True):
                # Al entrenar el modelo, en cada ejecucion de caracter se reinicializan las variables
                # y procesamiento y clasificacion de señal se realiza al finalizar todos los caracteres
                index = -1
                targets[row_actual-1] = 0
                targets[col_actual-1] = 0
                letter_trainer()
                return
            
            else:
                index = -1
                current_serie = 0
                currSerie = 0 # Variable que contiene el numero de serie actual

                # En el modo de evaluacion en tiempo real, se finaliza la ejecucion del caracteres (4 series)
                # esperamos a que finalice el hilo que permite obtener los datos del casco
                # y se comprueba la letra seleccionado realizando procesamiento y clasificacion de los datos
                process.join()
                letra = preprocess_and_classifier(csv_name_actual, mark_list, box)
                pred_label3.configure(text=letra)
                palabra = palabra + str(letra)
                pred_Text2.configure(state="normal")
                pred_Text2.insert(tkinter.INSERT, letra)    # Aunque sea incorrecta, insertamos la letra
                correct = messagebox.askyesnocancel("Success",
                                            f"¿Es correcta la letra seleccionada?",
                                            parent=root)
                if correct:

                    # En el caso de que sea correcta, se pregunta si se quiere seguir escribiendo
                    # si no es correcta, se repite la ejecucion
                    writing = messagebox.askyesno("Letra",
                                    f"¿Quiere seguir escribiendo?",
                                    parent=root)
                    if writing:

                        # En la evaluacion en tiempo real se modifica el uso de los csv
                        # se obtienen los datos del csv por la ejecucion de cada caracter
                        # por lo que van a aparecer tantos csv como letras deletreadas
                        csv_name_actual = "{}_{}.csv".format(csv_name, nletras)
                        mark_list.clear()
                        process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
                        nletras = nletras + 1
                        process.start()
                        root.after(2000)
                        change_color(0)

                elif correct == False:
                    mark_list.clear()
                    process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
                    process.start()
                    root.after(2000)
                    change_color(0)
                
                return
            
    if trainMode: # Separacion de fila y columna objetivo al flashear
        if current_serie != 0 and index != 0:
            while abs(List.index(row_actual) - List.index(col_actual)) < 2:
                shuffle(List)
    
    set(List[index], "CC") # Iluminacion de fila/columna
    currSerie = current_serie
    if (index >= 0):
        root.after(flash_time, unset, List[index], "CC") # Desiluminacion de fila/columna

def sendTimeStamp(index):
    global currSerie, targets, mark_list, outlet, testMode, stop_time, currSerie

    timestamp = local_clock()

    if testMode:
        mark_list[timestamp] = index    # Almacenamos timestamp y fila o columna que se ha iluminado   
        outlet.push_sample([3], timestamp)  # Etiqueta incluida para sincronizar los estímulos
    else:
        outlet.push_sample([2 if targets[index-1] == 0 else 1], timestamp)  # Etiqueta incluida para sincronizar los estímulos. 1 = Target, 2 = NoTarget
    
    root.after(stop_time, change_color, currSerie)  # Tiempo entre intensificaciones



def set(index, type):
    global flash_time, box
    font_size = 28
    font_type = 'Times New Roman'

    if (type == 'CC'):

        if (index < 7): # Iluminación de fila
            index1 = (index - 1) * 6 + 1

            for i in range(index1, index1 + 6):
                box[i].config(fg="white", font=(font_type, font_size, 'bold'))

            print("row: ", index)

        else:   # Iluminación de columna
            index1 = index - 6
            for i in range(index1, index1 + 31, 6):
                box[i].config(fg="white", font=(font_type, font_size, 'bold'))
                
            print("col: ", index1)

    elif (type == "caracter"):  # Iluminación de carácter antes de comenzar su ejecución
        box[index].config(background="blue", font=(font_type, font_size, 'bold'))
        print("caracter iluminado: {}".format(index))


def unset(index, type):
    global stop_time, trainMode, box, List, targets, currSerie, row_actual,col_actual
    font_size = 20
    font_type = 'Times New Roman'

    if (type == "CC"):

        if (index < 7): # Desiluminar fila
            index1 = (index - 1) * 6 + 1
            for i in range(index1, index1 + 6):
                box[i].config(fg="grey", font=(font_type, font_size))

            print("row ", index, " returned")

        else:   # Desiluminar columna
            index1 = index - 6
            for i in range(index1, index1 + 31, 6):
                box[i].config(fg="grey", font=(font_type, font_size))
                
            print("col ", index1, " returned")
        
        sendTimeStamp(index) # Envío de marca de tiempo mediante streams LSL
        
    elif (type == "caracter"):
        box[index].config(background="black", font=(font_type, font_size))
        if trainMode:
            #   Obtener fila y columna del carácter objetivo
            row = int((index - 1) / 6) + 1 
            col = (index - 1) % 6 + 7
            shuffle(List)
            #   Separación de iluminación entre fila/columna objetivo   
            while abs(List.index(row) - List.index(col)) < 2 or List.index(row) == 1 or List.index(col) == 1:
                shuffle(List)
            targets[row-1] = 1  # Fila objetivo
            targets[col-1] = 1  # Columna objetivo
            row_actual = row
            col_actual = col
            change_color(0)
        else:
            shuffle(List)


def letter_trainer():
    global current_index, train_text, trainMode
    global start_simulation, experiment_name

    if (start_simulation == False or trainMode == False):
        return
    
    if (current_index + 1 < len(train_text)):
        current_index += 1
        letter = train_text[current_index].upper()
        pred_label3.configure(text=letter)
        if (letter.isalpha()):  # Cálculo de posición de símbolo en la matriz
            box_num = ord(letter) - ord('A') + 1
        elif (letter.isnumeric()):
            box_num = ord(letter) - ord('0') + 1 + 26
        else:
            box_num = 36
        set(box_num, "caracter")
        root.after(3000, unset, box_num, "caracter")
    else:
        # Al finalizar todos los caracteres, se realiza el preprocesamiento y entrenamiento del modelo
        # Puede ser realizado también con train_signal.py
        current_index = -1
        state_controller()
        classify = messagebox.askyesno("Success", f"File {experiment_name} created!\nWould you like to preprocess data and train classifier?",
                                       parent=root)
        if classify is True:
            print("Doing Preprocess..........")  
            preprocess_and_train(csv_name)


def state_controller():
    global start_simulation, index, current_index, start_time_ts, start_time_f
    global stop_time_ts, stop_time_f, eeg_thread, testMode, trainMode

    if testMode == True and existe_clasificador() == False:
        start_simulation = False
        messagebox.showinfo("Alert", "No classifier trained!!!")
        return False
    if (start_simulation == False):
        start_time_ts = tm.time()
        start_time_f = tm.strftime("%H:%M:%S %p")
        start_simulation = True
        button.configure(text='Stop')
    else:
        start_simulation = False
        stop_time_ts = tm.time()
        stop_time_f = tm.strftime("%H:%M:%S %p")
        
        starttime.configure(text=start_time_f)
        stoptime.configure(text=stop_time_f)
        
        if trainMode:
            unset(List[index], "CC")
        index = -1
        current_index = -1
        button.configure(text='Start')


def start_speller():
    global start_simulation, index, start_time_ts, stop_time_ts, experiment_name, csv_name, csv_name_actual
    global start_time_f, stop_time_f, nletras, time, process
    global trainMode, train_text, current_index, info, outlet

    if trainMode is False and testMode is False:
        messagebox.showinfo("Alert", "Please select a mode!")
        return
    
    train_text = pred_Text2.get(1.0, "end-1c")
    if (trainMode is True and train_text is ""):
        messagebox.showinfo("Alert", "Please enter text for training!")
    else:
        if state_controller() == False:
            return
        experiment_name = pred_Text1.get(1.0, "end-1c")
        if experiment_name == "":
            experiment_name = "exp_{}".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

        now = datetime.now()
        if trainMode: 
            csv_name = "experiments/{}/records/record_{}.csv".format(experiment_name, now.strftime("%d-%m-%Y"))
            # Tiempo de captura de datos para entrenamiento de modelo
            time = (((12 * flash_time + 11 * stop_time) * seriesxchar + (seriesxchar - 1) * stop_time) * len(train_text))/ 1000 + 3.5 * len(train_text) 
            process = Thread(target=record_experiment, args=[experiment_name, time, ""])
        else:
            csv_name = "experiments/{}/records/record_{}".format(experiment_name, now.strftime("%d-%m-%Y"))
            # Tiempo de captura de datos para evaluación en tiempo real
            time = ((12 * flash_time + 11 * stop_time) * seriesxchar + (seriesxchar - 1) * stop_time) / 1000 + 2
            csv_name_actual = "{}_{}.csv".format(csv_name, nletras)
            process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
            nletras = nletras + 1
        
        process.start()
        
        
        info = StreamInfo('obci_eeg1', 'EEG', 1, 0, 'int32', 'objetivo12310')
        outlet = StreamOutlet(info)

        if(trainMode == False): # Tiempo de separación entre selección de caracteres en la evaluación en tiempo real
            root.after(2000)

        if not os.path.isdir('experiments/' + experiment_name):
            os.makedirs("experiments/{}/records".format(experiment_name))

        if (trainMode is True):
            current_index = -1
            letter_trainer()
        else:
            change_color(0)


root = Tk()
menubar = Menu(root)
root.config(menu=menubar, bg="#666666")

############# Modo de aplicación ##################### 

filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Mode', menu=filemenu)
filemenu.add_command(label='Word Training Mode', command=lambda: do_training())
filemenu.add_command(label='Testing Mode', command=lambda: do_testing())
filemenu.add_separator()
filemenu.add_command(label='Quit', command=sys.exit)

####################### Estructura de matriz #################################

box = dict() # Nuevo diccionario
import math

width = math.ceil(root.winfo_screenwidth() / 6) #Ancho de pantalla
height = math.ceil(root.winfo_screenheight() / 7.75) #Alto de pantalla
roots = Canvas(root, height=6 * height, width=6 * width, bg="#666666", highlightthickness=0, bd=0) #Canvas para lineas y texto de interfaz gráfica (6x6)
frame = {}

##======================Matriz 6x6 de P300 Speller=====================##
for r in range(1, 7):
    for c in range(1, 7):
        if (6 * (r - 1) + c <= 26):
            frame[6 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
            frame[6 * (r - 1) + c].pack_propagate(0)
            box[6 * (r - 1) + c] = Label(frame[6 * (r - 1) + c], text=chr(64 + 6 * (r - 1) + c), borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
            box[6 * (r - 1) + c].pack(fill="both", expand=True, side='left')
            frame[6 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)

        else:
            frame[6 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
            frame[6 * (r - 1) + c].pack_propagate(0)
            box[6 * (r - 1) + c] = Label(frame[6 * (r - 1) + c], text=6 * (r - 1) + c - 27, borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
            box[6 * (r - 1) + c].pack(fill="both", expand=True, side='left')
            frame[6 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)

roots.pack(fill="both", expand=True)

##===============================================================================##

##========================Barra inferior de datos=======================##
root7 = Canvas(root, bg="#666666", highlightthickness=0, bd=0)

pred_label0 = Label(root7, text="Experiment Name", fg="Black", font=("Times New Roman", 18), bg="#666666", width="14")
pred_label0.grid(row=0, column=0)

pred_Text1 = Text(root7, height=1, width=15, font=("Times New Roman", 18), state="disabled")
pred_Text1.grid(row=1, column=0)

pred_label1 = Label(root7, text="Predicted Text", fg="Black", font=("Times New Roman", 18), bg="#666666", width="12")
pred_label1.grid(row=0, column=1)

pred_Text2 = Text(root7, height=1, width=15, font=("Times New Roman", 18), state="disabled")
pred_Text2.grid(row=1, column=1)

pred_label2 = Label(root7, text="Predicted Letter", fg="Black", font=("Times New Roman", 18), bg="#666666", width="12",
                    padx="10")
pred_label2.grid(row=0, column=2)

pred_label3 = Label(root7, text="", fg="Black", font=("Times New Roman", 18), bg="#666666", padx="10", pady="10")
pred_label3.grid(row=1, column=2)

button = Button(root7, text='Start', width=10, height=1, fg="#666666", bg="#222222", font=("Times New Roman", 20),
                command=start_speller, activebackground="#666666", activeforeground="black")
button.grid(rowspan=2, row=0, column=3)

start_label = Label(root7, text='Start Time:', fg='black', font=("Times New Roman", 17), bg="#666666", padx="40")
start_label.grid(row=0, column=4)

starttime = Label(root7, text='00:00:00 00', fg='black', font=("Times New Roman", 17), bg="#666666", width="10")
starttime.grid(row=0, column=5)

stop_label = Label(root7, text='Stop Time: ', fg='black', font=("Times New Roman", 17), bg="#666666", padx="40")
stop_label.grid(row=1, column=4)

stoptime = Label(root7, text='00:00:00 00', fg='black', font=("Times New Roman", 17), bg="#666666", width="10")
stoptime.grid(row=1, column=5)

root7.pack()

root.title("P300 Speller Row/Columns")
root.mainloop()
