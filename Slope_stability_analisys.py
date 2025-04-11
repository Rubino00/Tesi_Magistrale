
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


# pd.set_option ("display.max_columns" , None)
df = pd.read_csv ("C:\\Users\\fraru\\OneDrive\\Desktop\\programmi\\Clone_python\\Tesi_magistrale\\Datastes\\slope_stability_dataset.csv")

def info_df (df:pd.DataFrame):
    print (df.info ())
    print (df.isnull ().sum ())
    print (df.head (30))

x_set = pd.get_dummies (df.drop (columns = ["Reinforcement Numeric" , "Factor of Safety (FS)"]) , columns = ["Reinforcement Type"])
y_set = df ["Factor of Safety (FS)"]

# dividiamo il nostro dataset in training e test
x_train , x_test , y_train , y_test = train_test_split (x_set , y_set , test_size = 0.5 , random_state = 42)

# scaliamo i valori di tutte le features del x_set
scaler = StandardScaler ()
x_train = scaler.fit_transform (x_train)
x_test = scaler.transform (x_test)

# per usare pytorch dobbiamo convertire il tutto in tensori
x_train = torch.tensor (x_train , dtype = torch.float32)
x_test = torch.tensor (x_test , dtype = torch.float32)
y_train = torch.tensor (np.array (y_train) , dtype = torch.float32).view (-1 , 1)
y_test = torch.tensor (y_test , dtype = torch.float32).view (-1 , 1)

# definiamo la classe della nostra rete neurale
class network (torch.nn.Module):

    def __init__ (self , input_dim , hidden_dim , output_dim):
        super (network , self).__init__ ()

        self.net = torch.nn.Sequential (torch.nn.Linear (input_dim , hidden_dim),
                                        torch.nn.ReLU (),
                                        torch.nn.Linear (hidden_dim , output_dim))

    def forward (self , x):
        return self.net (x)

# definiamo il numero di input e costruiamo la nostra rete
num_inputs = x_train.shape [1]
model = network (num_inputs , 64 , 1)

# costruiamo la nostra loss function
criterion = torch.nn.MSELoss ()

# costruiamo l'ottimizzatore
optimizer = torch.optim.Adam (model.parameters () , lr = 0.001)

num_epochs = 200
loss_list = []

for epoch in range (num_epochs):

    # attiviamo la modalità di training del modello
    model.train ()

    # annulliamo i vari gradienti della rete neurale
    optimizer.zero_grad ()

    # definiamo gli output
    output = model (x_train)

    # calocliamo la il valore della loss
    loss = criterion (output , y_train)

    # facciamo la backpropagation
    loss.backward ()

    # aggiorniamo i pesi
    optimizer.step ()

    # stampiamo ad ogni passaggio il valore di loss
    print (f"loss function val: {loss.item ()}")
    loss_list.append (loss.item ())

plt.plot (loss_list)
plt.show ()















