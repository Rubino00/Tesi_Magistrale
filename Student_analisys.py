
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# importiamo il dataset dei nostri studenti. COn questo dataset analizzeremo le performanche degli studenti in matematica
file_path = "C:\\Users\\fraru\OneDrive\Desktop\programmi\Clone_python\Tesi_magistrale\Datastes\Student_dataset\Math-Students.csv"
df = pd.read_csv (file_path)

# con questa funzione andiamo a stampare informazioni generali relative al dataset
def info_df (df: pd.DataFrame):
    df.info ()
    print (df.isnull ().sum ())
    print (df.head (30))

# con questa funzione andiamo a stampare informazioni più specifiche per ogni colonna
def descr_col (df: pd.DataFrame):
    for column in df.columns:
        print (f"colonna: {column}")
        print (df [column].describe ())
        print ()

# con questa funzione convertiamo in valori binari tutti i campi che possono prendere valore 0 o 1
# la variabole "map_dict" ci dice a quale voce corrisponde il valore 0 o 1
map_dict = {}
def binary_conv (col_name , new_col_name):
    unique_list = df [col_name].unique ()
    map = {unique_list [0]: 0 , unique_list [1]: 1}
    df [new_col_name] = df [col_name].map (map)
    map_dict [new_col_name] = map

# per una migliore visualizzazione del dizionario possiamo utilizzare questa funzione
def print_map_dict (map_dict):
    for key in map_dict:
        print (key)
        print (map_dict [key])
        print ()

# definiamo l'insieme di tutte le colonne che vogliamo convertire in formato binario
binary_conv_list = ["school" , "sex" , "address" , "Pstatus" , "famsup" , "activities" ,
                    "romantic" , "nursery" , "famsize"]

# convertiamo tutte le colonne presenti nella lista "binary_conv_list" in formato binario
for col_name in binary_conv_list:
    binary_conv (col_name , f"binary_{col_name}")

# definiamo il dataframe dei valori di input rimuovendo le colonne che non ci servono
x_set = df.drop (columns = ["Mjob" , "Fjob" , "reason" , "guardian" , "schoolsup" , "paid" , "higher" ,
                            "internet" , "school" , "sex" , "address" , "Pstatus" , "famsup" , "activities" ,
                            "romantic" , "nursery" , "famsize"])

# definiamo il dataframe dei valori di output: si vuole quindi effettuare una previsione sulla valutazione
# dello studente espressa nella colonna G3
y_set = df ["G3"]

# splittiamo il nostro datast in una parte per l'allenamento della rete neurale e una parte per la validazione
x_train , x_test , y_train , y_test = train_test_split (x_set , y_set , test_size = 0.25 , random_state = 42)

# scaliamo i valori. In questo caso è opportuno evitare di scalare i valori relativi alle colonne in formato binari
scaled_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                  'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2',
                  'G3', 'binary_school', 'binary_sex']

# scaliamo le features di input sia per il dataset di training sia per il dataset di validazione
scaler = StandardScaler ()
x_train [scaled_columns] = scaler.fit_transform (x_train [scaled_columns])
x_test [scaled_columns] = scaler.fit_transform (x_test [scaled_columns])

# per utilizzare torch e' opportuno convertire i dataframe in tensori
x_train = torch.tensor (data = np.array (x_train) , dtype = torch.float32)
x_test = torch.tensor (data = np.array (x_test) , dtype = torch.float32)
y_train = torch.tensor (data = np.array (y_train) , dtype = torch.float32).view (-1 , 1)
y_test = torch.tensor (data = np.array (y_test) , dtype = torch.float32).view (-1 , 1)

# scriavimo la nostra rete neurale
class network (torch.nn.Module):

    def __init__ (self , input_layer , hidden_layer , output_layer):
        super (network , self).__init__ ()

        self.net = torch.nn.Sequential (torch.nn.Linear (input_layer , hidden_layer),
                                        torch.nn.ReLU (),
                                        torch.nn.Linear (hidden_layer , output_layer))

    def forward (self , x):
        return self.net (x)

# per questo algoritmo utilizziamo come loss function la MSE e come optimizer Adam, costruiamo la nostra rete neurale
model = network (x_train.shape [1] , 64 , 1)
criterion = torch.nn.MSELoss ()
optimizer = torch.optim.Adam (model.parameters ())

# definiamo il processo di allenamento. Per ogni step prendiamo tutti il dataset e facciamo forward e backpropagation
epochs = 200
loss_list = []

for epoch in range (epochs):

    # settiamo la nostra rete in modalità training
    model.train ()

    # annulliamo il gradiente
    optimizer.zero_grad ()

    # eseguiamo la fase di forward e backpropagation
    output = model (x_train)
    loss = criterion (output , y_train)
    loss.backward ()
    optimizer.step ()

    # conserviamo nella lista "error_list" il valore di errore commesso per ogni step e mandiamolo a schermo
    print (f"valore di MSE: {loss.item ()}")
    loss_list.append (loss.item ())

# plottiamo il valore di errore per ogni epoca
plt.plot (loss_list)
plt.show ()










