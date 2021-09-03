# P300-speller-for-BCI

Framework that allows running a P300 speller application with row/column paradigm and region paradigm.
This framework handles all the phases of the bci cycle: from the acquisition of the signal by the BCI to the classification and execution of the result given by the classifier.

## Installation

1. Clone the repository
```
git clone https://github.com/rbnrtg/P300-speller-for-BCI.git
```

2. Access the cloned repository and install Python and required packages

- Python version 3.8.5 or more
- pip3 version 20.2.4 or more
- pip3 install -r requirements.txt

3. if all packages are installed, access P300 speller by rows/columns with:
```
python ventanaRC.py
```
and access P300 speller by regions with:
```
python ventanaRegion.py
```

## Files

- ventanaRC.py
Code that executes a P300 speller application by rows and columns. When accessing, the execution mode must be selected: to train the model or for real-time evaluation if there is a model already trained.

- ventanaRegion.py
Code that executes a P300 speller application by regions. When accessing, the execution mode must be selected: to train the model or for real-time evaluation if there is a model already trained.

- train_signal.ipynb
Jupyter Notebook where we can indicate which datasets we want to use to train the model and runs the model preprocessing and training phase. Once the model is trained, the trained classifier and the standardization object are imported through the Python pickle library to be used in the execution of the real-time evaluation of the P300 speller.
