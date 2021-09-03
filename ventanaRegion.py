import time as tm
import os
import sys
import string
from datetime import datetime
from threading import Thread
from tkinter import *
from tkinter import messagebox
from random import shuffle
from pylsl import StreamInfo, resolve_stream,StreamOutlet, local_clock
import math
from record import record_experiment #Importa función que captura y almacena los datos de la BCI
from preprocess_classification import preprocess_and_train, existe_clasificador, preprocess_and_classifierRegion

start_simulation = False
seriesxchar = 4
currSerie = 0
flash_time = 150
stop_time = 800
trainMode = False
testMode = False
trainRegiones = False
index = -1
experiment_name = ""
train_text = ""
csv_name = ""
csv_name_actual = ""
mark_list = dict()
row_actual = None
col_actual = None
nletras = 0 # Para llevar el nº de letra que se está registrando en el modo libre de P300 Speller
region = 0
alfabeto = string.ascii_uppercase
letraActual = ""
time = 0
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

List = [] # Lista para saber el orden en el que se iluminan filas y columnas
targets = [] # Contiene 5 posiciones que indican con 1 si la region es Target o 0 si es NoTarget
for i in range(1, 6):
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

# Cambio a mode de evaluacion en tiempo real
def do_testing():
    global trainMode, testMode, start_simulation

    if (start_simulation == True):
        messagebox.showwarning("Alert", "You Cannot switch modes currently!")
        return
    
    trainMode = False
    testMode = True
    pred_label1.configure(text="Predicted Text")
    if (pred_Text1["state"] == "disabled"):
        pred_Text1.configure(state="normal")
    pred_Text2.delete('1.0', END)
    pred_Text2.configure(state="disabled")
    pred_label2.configure(text="Predicted Letter")
    pred_label3.configure(text="")

# current_serie : serie por la que se encuentra una nivel en concreto (4 series)
def change_color(current_serie):
    global start_simulation, index, seriesxchar, train_text, current_index, csv_name_actual, nletras, root, process, time, palabra
    global trainMode, testMode, currSerie, mark_list, row_actual, col_actual, csv_name, region, letraActual, trainRegiones

    if (start_simulation == False):
        return

    index = (index + 1)
    if (index >= 5): # se comprueba si se ha completado ya la serie al completo

        current_serie += 1
        index = 0
        shuffle(List)
        if (current_serie >= seriesxchar):

            currSerie = 0 # Almacena el número de serie actual
            if (trainMode is True and trainRegiones is False):

                index = -1
                change_speller(2)  # Se cambia de nivel y se accede a la region correspondiente al caracter objetivo
                if letraActual.isnumeric():
                    if region == 6:
                        letra = (ord(letraActual) - ord('4')) + 1
                    else:
                        letra = (ord(letraActual)-ord('0')) + 3
                else:
                    letraRegion = ord('A') + (region-1)*6   # Se calcula la primera letra de la region objetivo
                    letra = ord(letraActual) - letraRegion + 1  # Se obtiene la posicion de la letra en la matriz

                print(targets)
                targets[row_actual-1] = 0
                targets[col_actual-1] = 0
                set(letra, "caracter")  # Se realiza la ejecucion del 2o nivel para el caracter
                trainRegiones = True    # Variable que indica si estamos en el 1o (False) o 2o (True) nivel
                root.after(3000, unset, letra, "caracter")
                return

            elif (trainMode and trainRegiones):

                # Cuando nos encontramos en el 2o nivel de entrenamiento de un caracter, reiniciamos variables de nuevo
                # se cambia el layout a nivel 1 y se comprueba se quedan mas caracteres para entrenar
                trainRegiones = False
                change_speller(1)
                index = -1
                print(targets)
                targets[row_actual-1] = 0
                targets[col_actual-1] = 0
                letter_trainer()
                
                return

            elif (testMode and trainRegiones is False):
                # Evaluacion en tiempo real
                # Acaba la ejecucion del primer nivel y esperamos a que el hilo de obtencion de datos del casco finalice
                # Por cada nivel finalizado, hay ue llamar al clasificador para que nos diga que region es la seleccionada
                index = -1
                process.join()
                fila, columna = preprocess_and_classifierRegion(csv_name_actual, mark_list)
                isRegion(fila, columna) # A partir de la fila y columna, obtengo la region
                change_speller(2)
                correct = messagebox.askyesnocancel("Success",
                                            f"¿Es correcta la región seleccionada?",
                                            parent=root)
                if correct:
                    # Si la region es la correcta, se accede a la evaluacion en tiempo real de 2o nivel (6 caracteres)
                    # Se crea nuevo nombre para csv y reinicializamos variables e hilo para obtener datos de la BCI
                    csv_name_actual = "{}_{}-{}.csv".format(csv_name, nletras, 2)
                    mark_list.clear()
                    trainRegiones = True # Indica que va a comenzar la ejecucion del 2o nivel
                    process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
                    process.start()
                    root.after(2000)    # Esperamos 2 segundos entre ejecuciones
                    change_color(0)
                
                elif correct == False:
                    # Si la region no es correcta, se vuelve a ejecutar el mismo nivel para intentar obtener la region correcta
                    mark_list.clear()
                    change_speller(1)
                    process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
                    process.start()
                    root.after(2000)
                    change_color(0)
                
                current_serie = 0
                currSerie = 0
                return
            
            elif (testMode and trainRegiones):
                # Finaliza la ejecucion del 2o nivel
                # Se espera a que termine la ejecucion del hilo ue obtiene los datos y se pasa a procesar y clasificar los datos obtenidos
                # Se obtiene la region seleccionada
                index = -1
                process.join()
                fila, columna = preprocess_and_classifierRegion(csv_name_actual, mark_list)
                isRegion(fila, columna)
                letra = str(box[region].cget("text"))
                pred_label3.configure(text=letra)
                palabra = palabra + str(letra)
                pred_Text2.configure(state="normal")
                pred_Text2.insert(INSERT, letra)
                correct = messagebox.askyesnocancel("Success",
                                            f"¿Es correcto el carácter seleccionado?",
                                            parent=root)

                if correct:
                    # Si el caracter es el correcta, se pregunta si se quiere seguir escribiendo
                    change_speller(1)
                    writing = messagebox.askyesno("Letra",
                                    f"¿Quiere seguir escribiendo?",
                                    parent=root)
                    if writing:
                        # Si se decide a seguir escribiendo, se realiza una nueva ejecucion con el nivel 1
                        trainRegiones = False
                        nletras = nletras + 1
                        csv_name_actual = "{}_{}-{}.csv".format(csv_name, nletras, 1)
                        mark_list.clear()
                        process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
                        process.start()
                        root.after(2000)
                        change_color(0)

                else:
                    # Si el caracter seleccionado no es el correcto, se vuelve a realizar la ejecucion del 2o nivel
                    csv_name_actual = "{}_{}-{}.csv".format(csv_name, nletras, 2)
                    mark_list.clear()
                    process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
                    process.start()
                    root.after(2000)
                    change_color(0)

                return
                
    if trainMode:
        if current_serie != 0 and index == 0: # En el caso de entrenamiento del modelo, separamos los flashes de la fila y columna objetivo
            while abs(List.index(row_actual) - List.index(col_actual)) < 2:
                shuffle(List)

    set(List[index], "CC")
    currSerie = current_serie
    if (index >= 0):
        root.after(flash_time, unset, List[index], "CC")

def sendTimeStamp(index):
    global currSerie, targets, mark_list, outlet, testMode, stop_time

    timestamp = local_clock()
    if testMode:
        mark_list[timestamp] = index
        print(timestamp)
        outlet.push_sample([3], timestamp)  # Enviamos etiqueta y timestamp por LSL
    else:
        outlet.push_sample([2 if targets[index-1] == 0 else 1], timestamp) # Enviamos etiqueta y timestamp por LSL
    
    root.after(stop_time, change_color, currSerie) # Tiempo entre intensificaciones



def set(index, type):

    global flash_time, box, trainRegiones
    font_size = 28
    font_type = 'Times New Roman'

    if (type == 'CC'):

        if (index < 3): # Iluminacion de filas
            if index == 2:
                index1 = 4
            else:
                index1 = index
            if trainRegiones:
                for i in range(index1, index1 + 3):
                    box[i].config(fg="white", font=(font_type, 40, 'bold'))
            else:
                for i in range(index1, index1 + 3):
                    box[i].config(fg="white", font=(font_type, font_size, 'bold'))

            print("row: ", index)

        else:   # Iluminacion de columnas
            index1 = index - 2
            if trainRegiones: # Cuando nos encontramos en el 2o nivel, hacemos que las letras al iluminarse se vean mas grandes
                for i in range(index1, index1 + 4, 3):
                    box[i].config(fg="white", font=(font_type, 40, 'bold'))    
            else:
                for i in range(index1, index1 + 4, 3):
                    box[i].config(fg="white", font=(font_type, font_size, 'bold'))
            
            print("col: ", index1)

    elif (type == "caracter"):

        box[index].config(background="blue", font=(font_type, font_size, 'bold'))
        print("index received : {}".format(index))


def unset(index, type):
    
    global trainMode, box, List, targets, row_actual,col_actual
    font_size = 20
    font_type = 'Times New Roman'

    if (type == "CC"):
        if (index < 3): # Desiluminacion de filas
            if index == 2:
                index1 = 4
            else:
                index1 = index
            for i in range(index1, index1 + 3):
                box[i].config(fg="grey", font=(font_type, font_size))
            print("row ", index, " returned")
        else:   # Desiluminacion de columnas
            index1 = index - 2
            for i in range(index1, index1 + 4, 3):
                box[i].config(fg="grey", font=(font_type, font_size))
            print("col ", index1, " returned")
        
        sendTimeStamp(index)    # Se envia timestamo y etiqueta por los streams de LSL
        
    elif (type == "caracter"):

        box[index].config(background="black", font=(font_type, font_size))
        if trainMode:
            if index > 3:
                row = 2 # Calculamos fila objetivo
            else:
                row = 1
            if index < 4:
                col = index + 2 # Calculamos columna objetivo
            else:
                col = index - 1
            shuffle(List)
            # Separacion de iluminacion de fila y columna objetivo
            while abs(List.index(row) - List.index(col)) < 2 or List.index(row) == 1 or List.index(col) == 1: 
                shuffle(List)
            targets[row-1] = 1
            targets[col-1] = 1
            row_actual = row
            col_actual = col
            isRegion(row, col) # Calculamos la region a la que pertenece el caracter objetivo
            change_color(0)
        else:
            shuffle(List)   # Se mezclan de nuevo el orden de iluminacion

# A partir de fila y columna, se  devuelve la region interseccion entre ambas
def isRegion(row, col):
    global region
    print("row: " + str(row) + " col: " + str(col))
    if row == 1:
        if col == 3:
            region = 1
        elif col == 4:
            region = 2
        elif col == 5:
            region = 3
    elif row == 2:
        if col == 3:
            region = 4
        elif col == 4:
            region = 5
        elif col == 5:
            region = 6
    print("REGION: " + str(region))

    
def letter_trainer():
    global current_index, train_text, trainMode, letraActual
    global start_simulation, experiment_name, csv_name

    if (start_simulation == False or trainMode == False):
        return
    
    if (current_index + 1 < len(train_text)):
        current_index += 1
        letter = train_text[current_index].upper()
        letraActual = letter
        pred_label3.configure(text=letter)
        if (letter.isalpha()):
            box_num = math.trunc((ord(letter) - ord('A'))/6) + 1
            print("BOX: " + str(box_num))
        elif (letter.isnumeric()):
            box_num = math.trunc((ord(letter) - ord('0') + 26)/6 + 1)
        else:
            box_num = 36

        set(box_num, "caracter")
        root.after(3000, unset, box_num, "caracter")
    else:
        current_index = -1
        state_controller()
        classify = messagebox.askyesno("Success", f"File {experiment_name} created!\nWould you like to preprocess data and train classifier?",
                                       parent=root)
        if classify is True:
            print("Doing Preprocess..........")
            preprocess_and_train(csv_name)


def state_controller():
    global start_simulation, index, current_index, start_time_ts, start_time_f
    global stop_time_ts, stop_time_f, testMode, trainMode

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
    global start_simulation, index, start_time_ts, stop_time_ts, experiment_name, csv_name, csv_name_actual, process
    global start_time_f, stop_time_f, nletras, time
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
            time = ((((5 * flash_time + 4 * stop_time) * seriesxchar + (seriesxchar - 1) * stop_time) + ((5 * flash_time + 4 * stop_time) * seriesxchar + (seriesxchar - 1) * stop_time)) * len(train_text))/ 1000 + 8 * len(train_text)
            process = Thread(target=record_experiment, args=[experiment_name, time, ""])
        else:
            csv_name = "experiments/{}/records/record_{}".format(experiment_name, now.strftime("%d-%m-%Y"))
            time = ((5 * flash_time + 4 * stop_time) * seriesxchar + (seriesxchar - 1) * stop_time) / 1000 + 2
            csv_name_actual = "{}_{}-{}.csv".format(csv_name, nletras, 1)
            process = Thread(target=record_experiment, args=[experiment_name, time, csv_name_actual])
        
        process.start()

        info = StreamInfo('obci_eeg1', 'EEG', 1, 0, 'int32', 'objetivo12310')
        outlet = StreamOutlet(info)

        if(trainMode == False):
            root.after(2000)

        if not os.path.isdir('experiments/' + experiment_name):
            os.makedirs("experiments/{}/records".format(experiment_name))
        if (trainMode is True):
            current_index = -1
            letter_trainer()
        else:
            change_color(0)


# Método para cambiar entre los dos niveles existentes en P300 speller por regiones
# Nivel 1: 6 regiones de 6 caracteres. Nivel 2 : accede a los 6 caracteres de la variable 'region'
def change_speller(type):
    global region, alfabeto

    aux = 0
    n = 1
    if type == 1:
        number = 0
        for r in range(1, 3):
            for c in range(1, 4):

                frame[3 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
                frame[3 * (r - 1) + c].pack_propagate(0)  
                text = ""
                #n = 64 + 6 * (r - 1) + c
                if n >= 5:
                    for i in range(0,6):
                        if i <= 1 and n == 5:
                            text = text + chr(64 + n + i + aux * 5) + ' '
                        else:
                            text = text + str(number) + ' '
                            number = number + 1                    
                else: 
                    for i in range(0,6):
                        text = text + chr(64 + n + i + aux * 5) + ' '
                aux = aux + 1
                n = n + 1
                box[3 * (r - 1) + c] = Label(frame[3 * (r - 1) + c], text=text, borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
                box[3 * (r - 1) + c].pack(fill="both", expand=True, side='left')
                frame[3 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)
    elif type == 2:
        if region == 6:
            number = 4
        else:
            number = 0
        for r in range(1, 3):
            for c in range(1, 4):
                frame[3 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
                frame[3 * (r - 1) + c].pack_propagate(0) 
                text = ""
                #n = 64 + 6 * (r - 1) + c
                if region == 5:
                    if aux <= 1:
                        text = alfabeto[(region-1)*6 + aux]
                    else:
                        text = str(number)
                        number = number + 1  
                elif region == 6:
                    text = str(number)
                    number = number + 1
                else: 
                    print("REGION: " + str(region))
                    text = alfabeto[(region-1)*6 + aux]
                aux = aux + 1
                n = n + 1
                box[3 * (r - 1) + c] = Label(frame[3 * (r - 1) + c], text=text, borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
                box[3 * (r - 1) + c].pack(fill="both", expand=True, side='left')
                frame[3 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)



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

############# Estrctura de layout ##################### 

box = dict() # Nuevo diccionario

width = math.ceil(root.winfo_screenwidth() / 3) #Ancho de pantalla
height = math.ceil(root.winfo_screenheight() / 2.5) #Alto de pantalla
roots = Canvas(root, height=2 * height, width= 3 * width , bg="#666666", highlightthickness=0, bd=0) #Canvas para lineas y texto de interfaz gráfica
frame = {}

##======================Layout formado por 6 regiones de 6 caracteres=====================##

aux = 0
n = 1
number = 0

for r in range(1, 3):
    for c in range(1, 4):

        frame[3 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
        frame[3 * (r - 1) + c].pack_propagate(0) 
        text = ""
        #n = 64 + 6 * (r - 1) + c
        if n >= 5:
            for i in range(0,6):
                if i <= 1 and n == 5:
                    text = text + chr(64 + n + i + aux * 5) + ' '
                else:
                    text = text + str(number) + ' '
                    number = number + 1
                    
        else: 
            for i in range(0,6):
                text = text + chr(64 + n + i + aux * 5) + ' '
        aux = aux + 1
        n = n + 1
        box[3 * (r - 1) + c] = Label(frame[3 * (r - 1) + c], text=text, borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
        box[3 * (r - 1) + c].pack(fill="both", expand=True, side='left')
        frame[3 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)


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
root.title("P300 Speller Regions")
root.mainloop()