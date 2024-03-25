import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import tensorflow as tf

import torch
from torch import nn
from torch.optim import SGD
from torch.nn import MSELoss
import torch.nn.functional as F


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from scikeras.wrappers import KerasRegressor
from keras.initializers import RandomUniform
from keras import regularizers



import random
import itertools


from scipy.stats import uniform, randint, loguniform
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputRegressor


from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import random
import itertools



#for the training set, the test set has a different format (doesnt have the targets)
def cup_transform(cup_csv):
    #the first 7 rows are useless
    cup=pd.read_csv(cup_csv, sep=',', header=None, skipinitialspace=True, skiprows=7)
    
    #here we drop the first column which contains the id of the record (not useful)
    cup = cup.iloc[:, 1:]


    #we split the target variable from the other data
    y = cup.iloc[:,10:]
    X = cup.iloc[:,0:10]
    
    
    #data transformed in numpy arrays for efficiency
    y=np.array(y)
    X=np.array(X)
    
    return X, y


def cup_transform_test(cup_csv):
    #the first 7 rows are useless
    cup=pd.read_csv(cup_csv, sep=',', header=None, skipinitialspace=True, skiprows=7)
    
    #here we drop the first column which contains the id of the record (not useful)
    cup = cup.iloc[:, 1:]


    #we split the target variable from the other data
    #y = cup.iloc[:,10:]
    #X = cup.iloc[:,0:10]
    X = cup
    
    #data transformed in numpy arrays for efficiency
    #y=np.array(y)
    X=np.array(X)
    
    return X



#for the n_best losses and their combinations
def insert_in_best(best_losses, best_combinations, value, combination):
    index = 0
    for i in range(len(best_losses)):
        if value > best_losses[index]:
            index+=1
    best_losses.insert(index, value)
    best_combinations.insert(index, combination)
    best_losses.pop()
    best_combinations.pop()
    


def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=1)))



def create_keras_regressor(units_activations, learning_rate, momentum, l2, x_train):
    model = Sequential()
    for i in range(len(units_activations[0])):
        if i == 0:
                  
                
            model.add(Dense(units_activations[0][i], input_dim=x_train.shape[1],
                            kernel_regularizer=regularizers.l2(l2), activation=units_activations[1][i]))    
        else:
           
            model.add(Dense(units_activations[0][i],
                        kernel_regularizer=regularizers.l2(l2), activation=units_activations[1][i]))

    model.add(Dense(3, activation='linear'))  # 3 output neurons for 3 target variables
    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics = [mean_euclidean_error])
    return model



#randomized search for a first look on a few combinations of hyperparameters
#it plots the accuracy and loss for every combination and return the n_best loss and their combinations of hyperparameters
def randomized_search_keras_regressor(X_train, Y_train, param_dict, number_of_combinations, n_best):

    n_splits = 5  #number of splits for k-fold

    random_combinations = random.sample(list(itertools.product(*param_dict.values())), number_of_combinations)

    best_combinations = [None] * n_best #where we store the n_best combinations of hyperparameters
    best_losses = [float('inf')] * n_best #where we store the n_best losses

    for combo in random_combinations:
        param_dict = dict(zip(param_dict.keys(), combo))
        print("Testing with hyperparameters:", param_dict)

        #unpack hyperparameters
        
        units_activations = param_dict['units_activations']
        learning_rate = param_dict['learning_rate']
        momentum = param_dict['momentum']
        l2 = param_dict['l2']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']
        
        #build the model
        keras_model = KerasRegressor(model=create_keras_regressor, x_train = X_train, units_activations = units_activations, 
                                      learning_rate=learning_rate,
                                      momentum=momentum, l2=l2, optimizer='sgd', verbose = 0)

        # where we append the stats for each run of the cross validation
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []
        

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
            print(f"Fold {fold + 1}:")

            #split the data
            x_train, y_train = X_train[train_indices], Y_train[train_indices]
            x_val, y_val = X_train[val_indices], Y_train[val_indices]

 

            #fit the model
            keras_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs = epochs, batch_size = batch_size)
    
            # history
            train_loss = keras_model.history_['mean_euclidean_error']
            val_loss = keras_model.history_['val_mean_euclidean_error']
            

            print(f"Training Loss: {train_loss[-1]}")
            print(f"Validation Loss: {val_loss[-1]}")

            #append the stats for the this run
            train_losses.append(train_loss)         
            val_losses.append(val_loss)

        #calculate mean loss and accuracy over all folds
        train_mean_loss = [sum(losses) / len(losses) for losses in zip(*train_losses)]
        val_mean_loss = [sum(losses) / len(losses) for losses in zip(*val_losses)]

        #take the last validation loss, we will use it for the selection of the best losses
        last_epoch_val_mean_loss = val_mean_loss[-1]

        #update the best losses and best combinations
        insert_in_best(best_losses, best_combinations, last_epoch_val_mean_loss, param_dict)


        #plot mean training and validation loss
        plt.figure(figsize=(12, 6))
        plt.plot(train_mean_loss, label='Train')
        plt.plot(val_mean_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        #plot standard deviation for training and validation loss
        train_loss_std = np.std(train_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)

        plt.fill_between(range(epochs), np.array(train_mean_loss) - train_loss_std, np.array(train_mean_loss) + train_loss_std, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), np.array(val_mean_loss) - val_loss_std, np.array(val_mean_loss) + val_loss_std, alpha=0.3, label='Validation Std Dev')

        #plt.ylim(bottom=0, top=last_epoch_val_mean_loss * 5) # otherwise we cant see the zigzagin and the std
        plt.legend()
        plt.tight_layout()
        plt.show()


    #best parameters and results
    
    for i in range(len(best_losses)):
        print(i, best_losses[i], best_combinations[i])
        



def grid_search_keras_regressor(param_grid, X_train, y_train):
    model = KerasRegressor(model=create_keras_regressor, optimizer = 'sgd', x_train=X_train)



    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=KFold(n_splits=5),
        n_jobs=-1,
        verbose=3,
        return_train_score=True
    )


    grid.fit(X_train, y_train, verbose=1)
    print(grid.best_params_)
    return grid.best_params_, grid.best_estimator_



def plot_best_keras_regressor_validation(best_estimator, epochs, X_train, Y_train):

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    n_splits = 5

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

        #split the data
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

    

        #fit the model
        best_estimator.fit(x_train, y_train, validation_data=(x_val, y_val), verbose = 0)

        # history
        train_loss = best_estimator.history_['mean_euclidean_error']
        val_loss = best_estimator.history_['val_mean_euclidean_error']


        print(f"Training Loss: {train_loss[-1]}")
        print(f"Validation Loss: {val_loss[-1]}")

        #append the stats for the this run
        train_losses.append(train_loss)         
        val_losses.append(val_loss)

    #calculate mean loss and accuracy over all folds
    train_mean_loss = [sum(losses) / len(losses) for losses in zip(*train_losses)]
    val_mean_loss = [sum(losses) / len(losses) for losses in zip(*val_losses)]

    #take the last validation loss, we will use it for the selection of the best losses
    last_epoch_val_mean_loss = val_mean_loss[-1]


    #plot mean training and validation accuracy

    #plot mean training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_mean_loss, label='Train')
    plt.plot(val_mean_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    #plot standard deviation for training and validation loss
    train_loss_std = np.std(train_losses, axis=0)
    val_loss_std = np.std(val_losses, axis=0)

    plt.fill_between(range(epochs), np.array(train_mean_loss) - train_loss_std, np.array(train_mean_loss) + train_loss_std, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), np.array(val_mean_loss) - val_loss_std, np.array(val_mean_loss) + val_loss_std, alpha=0.3, label='Validation Std Dev')

    plt.ylim(bottom=0, top=17.5) # otherwise we cant see the zigzagin and the std
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(last_epoch_val_mean_loss)





def plot_best_keras_regressor_test(X_train, y_train, X_test, y_test, best_estimator):
    
    best_estimator.fit(X_train, y_train, validation_data=(X_test, y_test))
    y_pred = best_estimator.predict(X_test)
    
    #plot training & validation loss values
    plt.plot(best_estimator.history_['mean_euclidean_error'])
    plt.plot(best_estimator.history_['val_mean_euclidean_error'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.ylim(bottom=0, top=17.5) # otherwise we cant see the zigzagin and the std
    plt.show()
    
    print(best_estimator.history_['val_mean_euclidean_error'][-1])










###################################################################################################################################################
#PYTORCH############################################################################################################################################
###################################################################################################################################################













class Torch_Regressor(nn.Module):
    def __init__(self, input_size, output_size, layers_data, weight_init=torch.nn.init.xavier_uniform_):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.input_size=input_size
        for n_units, activation in layers_data:
            self.layers.append(nn.Linear(input_size, n_units))
            input_size=n_units
            if activation is not None:
                self.layers.append(activation)
                
        self.output = nn.Linear(input_size, output_size)

        #glorot weights initilization
        self.layers.apply(self._init_weights)
        self.output.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
            
        x=self.output(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#layers_data is a list containing tuples
#the first position of the tuple contains n_units of the layer, the second one the activation functions
 

    
def create_torch_regressor(input_size, output_size, layers_data, learning_rate, momentum, l2):
    model = Torch_Regressor(input_size, output_size, layers_data)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2)
    criterion = nn.MSELoss()
    return model, optimizer, criterion




#with this function we perform randomized search by using repeated k fold
def randomized_search_pytorch_reg(X_train, Y_train, param_dict, number_of_combinations, n_best, seed=0):
    torch.manual_seed(seed)

    n_splits = 7

    random.seed(0)
    random_combinations = random.sample(list(itertools.product(*param_dict.values())), number_of_combinations)

    best_combinations = [None] * n_best
    best_mee = [float('inf')] * n_best

    for combo in random_combinations:
        param_dict = dict(zip(param_dict.keys(), combo))
        print("Testing with hyperparameters:", param_dict)

        layers_data = param_dict['layers_data']
        learning_rate = param_dict['learning_rate']
        momentum = param_dict['momentum']
        l2 = param_dict['l2']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']

        all_train_mee = []
        all_train_losses = []
        all_val_mee = []
        all_val_losses = []
        
        
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(rkf.split(X_train, Y_train)):
            print(f"Fold {fold + 1}:")

            x_train, y_train = X_train[train_indices], Y_train[train_indices]
            x_val, y_val = X_train[val_indices], Y_train[val_indices]

            x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            
            train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
            val_dataset = CustomDataset(x_val_tensor, y_val_tensor)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            input_size = x_train.shape[1]
            output_size=y_train.shape[1]
            
            # Build the model
            model, optimizer, criterion = create_torch_regressor(input_size, output_size, layers_data, learning_rate, momentum, l2)

            training_epoch_loss = []
            validation_epoch_loss = []
            training_epoch_mee = []
            validation_epoch_mee = []

            # Fit the model
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                running_distance = 0.0
                running_total= 0
                
                for i, (inputs, labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    
                    #compute euclidean error
                    running_distance+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()
                    running_total += labels.size(0)
                
                
                # Calculate the training loss and mee
                train_mee = running_distance / running_total
                train_loss = running_loss / len(train_dataloader.dataset)
                training_epoch_loss.append(train_loss)
                training_epoch_mee.append(train_mee)

                
                # Validation
                model.eval()
                val_loss = 0.0
                distances = 0.0
                total = 0

                with torch.no_grad():
                    for data in val_dataloader:
                        inputs, labels = data
                        outputs = model(inputs)
                        
                        val_loss += criterion(outputs, labels).item() * labels.size(0)
                        total += labels.size(0)
                        distances+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()
                 
                # Calculate the validation loss
                val_mee = distances / total
                val_loss /= len(val_dataloader.dataset)
                validation_epoch_mee.append(val_mee)
                validation_epoch_loss.append(val_loss)

                
           
            print(f"Training Loss: {training_epoch_loss[-1]}")
            print(f"Validation Loss: {validation_epoch_loss[-1]}")
            print(f"Training MEE: {validation_epoch_mee[-1]}")
            print(f"Validation MEE: {validation_epoch_mee[-1]}")
     
            # Store results for this fold
            
            all_train_losses.append(training_epoch_loss)
            all_val_losses.append(validation_epoch_loss)
            all_train_mee.append(training_epoch_mee)
            all_val_mee.append(validation_epoch_mee)
            
           

        # Calculate mean and standard deviation across folds
        mean_train_loss = np.mean(all_train_losses, axis=0)
        std_train_loss = np.std(all_train_losses, axis=0)
        mean_train_mee=np.mean(all_train_mee, axis=0)
        std_train_mee = np.std(all_train_mee, axis=0)

        
        mean_val_loss = np.mean(all_val_losses, axis=0)
        std_val_loss = np.std(all_val_losses, axis=0)
        mean_val_mee=np.mean(all_val_mee, axis=0)
        std_val_mee=np.std(all_val_mee, axis=0)

        # Plot mean training and validation loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(mean_train_loss, label='Train')
        plt.plot(mean_val_loss, label='Validation')
        plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        
        # Plot mean training and validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(mean_train_mee, label='Train')
        plt.plot(mean_val_mee, label='Validation')
        plt.fill_between(range(epochs), mean_train_mee - std_train_mee, mean_train_mee + std_train_mee, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_mee - std_val_mee, mean_val_mee + std_val_mee, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('MEE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


        # Insert the best combination and loss into the list
        insert_in_best(best_mee, best_combinations, mean_val_mee[-1], param_dict)

    # Print the best combinations and their losses
    print("\nBest Combinations:")
    for i in range(n_best):
        print(f"{i+1}. Hyperparameters: {best_combinations[i]}, Mean Validation MEE: {best_mee[i]:.4f}")






def grid_search_pytorch_reg(X_train, Y_train, param_dict, n_best, seed=0):
    torch.manual_seed(seed)
    n_splits = 7

    combinations = list(itertools.product(*param_dict.values()))
    
    best_combinations = [None] * n_best
    best_mee = [float('inf')] * n_best

    for combo in combinations:
        param_dict = dict(zip(param_dict.keys(), combo))
        print("Testing with hyperparameters:", param_dict)

        layers_data = param_dict['layers_data']
        learning_rate = param_dict['learning_rate']
        momentum = param_dict['momentum']
        l2 = param_dict['l2']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']

        all_train_mee = []
        all_train_losses = []
        all_val_mee = []
        all_val_losses = []

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(rkf.split(X_train, Y_train)):
            print(f"Fold {fold + 1}:")

            x_train, y_train = X_train[train_indices], Y_train[train_indices]
            x_val, y_val = X_train[val_indices], Y_train[val_indices]

            x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            
            train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
            val_dataset = CustomDataset(x_val_tensor, y_val_tensor)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            input_size=x_train.shape[1]
            output_size=y_train.shape[1]
            
            # Build the model
            model, optimizer, criterion = create_torch_regressor(input_size, output_size, layers_data, learning_rate, momentum, l2)

            training_epoch_loss = []
            validation_epoch_loss = []
            training_epoch_mee = []
            validation_epoch_mee = []

            # Fit the model
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                running_distance = 0.0
                running_total= 0
                
                for i, (inputs, labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    
                    #compute euclidean error
                    running_distance+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()
                    running_total += labels.size(0)
                
                #print(running_distance)
                #print(running_total)
                # Calculate the training loss and mee
                train_mee = running_distance / running_total
                train_loss = running_loss / len(train_dataloader.dataset)
                training_epoch_loss.append(train_loss)
                training_epoch_mee.append(train_mee)

                
                # Validation
                model.eval()
                val_loss = 0.0
                distances = 0.0
                total = 0

                with torch.no_grad():
                    for data in val_dataloader:
                        inputs, labels = data
                        outputs = model(inputs)
                        
                        val_loss += criterion(outputs, labels).item() * labels.size(0)
                        total += labels.size(0)
                        distances+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()
                 
                # Calculate the validation loss
                val_mee = distances / total
                val_loss /= len(val_dataloader.dataset)
                validation_epoch_mee.append(val_mee)
                validation_epoch_loss.append(val_loss)

                
           
            print(f"Training Loss: {training_epoch_loss[-1]}")
            print(f"Validation Loss: {validation_epoch_loss[-1]}")
            print(f"Training MEE: {validation_epoch_mee[-1]}")
            print(f"Validation MEE: {validation_epoch_mee[-1]}")
     
            # Store results for this fold
            
            all_train_losses.append(training_epoch_loss)
            all_val_losses.append(validation_epoch_loss)
            all_train_mee.append(training_epoch_mee)
            all_val_mee.append(validation_epoch_mee)
            
           

        # Calculate mean and standard deviation across folds
        mean_train_loss = np.mean(all_train_losses, axis=0)
        std_train_loss = np.std(all_train_losses, axis=0)
        mean_train_mee=np.mean(all_train_mee, axis=0)
        std_train_mee = np.std(all_train_mee, axis=0)

        
        mean_val_loss = np.mean(all_val_losses, axis=0)
        std_val_loss = np.std(all_val_losses, axis=0)
        mean_val_mee=np.mean(all_val_mee, axis=0)
        std_val_mee=np.std(all_val_mee, axis=0)

        # Plot mean training and validation loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(mean_train_loss, label='Train')
        plt.plot(mean_val_loss, label='Validation')
        plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        
        
        # Plot mean training and validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(mean_train_mee, label='Train')
        plt.plot(mean_val_mee, label='Validation')
        plt.fill_between(range(epochs), mean_train_mee - std_train_mee, mean_train_mee + std_train_mee, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_mee - std_val_mee, mean_val_mee + std_val_mee, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('MEE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        # Insert the best combination and loss into the list
        insert_in_best(best_mee, best_combinations, mean_val_mee[-1], param_dict)

    # Print the best combinations and their losses
    to_return=[]
    print("\nBest Combinations:")
    for i in range(n_best):
        print(f"{i+1}. Hyperparameters: {best_combinations[i]}, Mean Validation MEE: {best_mee[i]:.4f}")
        to_return.append(best_combinations[i])

    return to_return



def plot_best_torch_regressor_validation(best_estimator, X_train, Y_train, n_splits, seed=0):
    torch.manual_seed(seed)
    
    all_train_mee = []
    all_train_losses = []
    all_val_mee = []
    all_val_losses = []
    
    layers_data = best_estimator['layers_data']
    learning_rate = best_estimator['learning_rate']
    momentum = best_estimator['momentum']
    l2 = best_estimator['l2']
    batch_size = best_estimator['batch_size']
    epochs = best_estimator['epochs']

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(rkf.split(X_train, Y_train)):

        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
        val_dataset = CustomDataset(x_val_tensor, y_val_tensor)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        input_size = x_train.shape[1]
        output_size=y_train.shape[1]
            
        # Build the model
        model, optimizer, criterion = create_torch_regressor(input_size, output_size, layers_data, learning_rate, momentum, l2)

        training_epoch_loss = []
        validation_epoch_loss = []
        training_epoch_mee = []
        validation_epoch_mee = []

        # Fit the model
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_distance = 0.0
            running_total= 0

            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                #compute euclidean error
                running_distance+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()
                running_total += labels.size(0)

            #print(running_distance)
            #print(running_total)
            # Calculate the training loss and mee
            train_mee = running_distance / running_total
            train_loss = running_loss / len(train_dataloader.dataset)
            training_epoch_loss.append(train_loss)
            training_epoch_mee.append(train_mee)


            # Validation
            model.eval()
            val_loss = 0.0
            distances = 0.0
            total = 0

            with torch.no_grad():
                for data in val_dataloader:
                    inputs, labels = data
                    outputs = model(inputs)

                    val_loss += criterion(outputs, labels).item() * labels.size(0)
                    total += labels.size(0)
                    distances+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()

            # Calculate the validation loss
            val_mee = distances / total
            val_loss /= len(val_dataloader.dataset)
            validation_epoch_mee.append(val_mee)
            validation_epoch_loss.append(val_loss)



        print(f"Training Loss: {training_epoch_loss[-1]}")
        print(f"Validation Loss: {validation_epoch_loss[-1]}")
        print(f"Training MEE: {validation_epoch_mee[-1]}")
        print(f"Validation MEE: {validation_epoch_mee[-1]}")

        # Store results for this fold

        all_train_losses.append(training_epoch_loss)
        all_val_losses.append(validation_epoch_loss)
        all_train_mee.append(training_epoch_mee)
        all_val_mee.append(validation_epoch_mee)



    # Calculate mean and standard deviation across folds
    mean_train_loss = np.mean(all_train_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)
    mean_train_mee=np.mean(all_train_mee, axis=0)
    std_train_mee = np.std(all_train_mee, axis=0)


    mean_val_loss = np.mean(all_val_losses, axis=0)
    std_val_loss = np.std(all_val_losses, axis=0)
    mean_val_mee=np.mean(all_val_mee, axis=0)
    std_val_mee=np.std(all_val_mee, axis=0)

    # Plot mean training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.plot(mean_train_loss, label='Train')
    plt.plot(mean_val_loss, label='Validation')
    plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, label='Validation Std Dev')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    # Plot mean training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(mean_train_mee, label='Train')
    plt.plot(mean_val_mee, label='Validation')
    plt.fill_between(range(epochs), mean_train_mee - std_train_mee, mean_train_mee + std_train_mee, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), mean_val_mee - std_val_mee, mean_val_mee + std_val_mee, alpha=0.3, label='Validation Std Dev')
    plt.xlabel('Epoch')
    plt.ylabel('MEE')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    print(f"Hyperparameters: {best_estimator}, Mean Validation Loss: {mean_val_loss[-1]}, std Validation Loss: {std_val_loss[-1]}, Mean MEE: {mean_val_mee[-1]}, std MEE: {std_val_mee[-1]}")
    print(f", Mean training MEE: {mean_train_mee[-1]}, std training MEE: {std_train_mee[-1]}")



    
    




 
def retraining_pytorch_reg(final_parameters, X, y, X_val, y_val, seed=0):
    torch.manual_seed(seed)

    def torch_format(X, y):
        X=torch.FloatTensor(X)
        y=torch.FloatTensor(y)
        return X, y

    X_train, y_train=torch_format(X, y)
    X_test, y_test=torch_format(X_val, y_val)
    input_size = X_train.shape[1]
    output_size=y_train.shape[1]
            
    
    
    layers_data = final_parameters['layers_data']
    learning_rate = final_parameters['learning_rate']
    momentum = final_parameters['momentum']
    l2 = final_parameters['l2']
    batch_size = final_parameters['batch_size']
    epochs = final_parameters['epochs']
    
    
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #we define the model
    # Build the model
    model, optimizer, criterion = create_torch_regressor(input_size, output_size, layers_data, learning_rate, momentum, l2)
    
    training_epoch_loss = []
    validation_epoch_loss = []
    training_epoch_mee = []
    validation_epoch_mee = []

    # Fit the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_distance = 0.0
        running_total= 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            #compute euclidean error
            running_distance+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()
            running_total += labels.size(0)

        
        # Calculate the training loss and mee
        train_mee = running_distance / running_total
        train_loss = running_loss / len(train_dataloader.dataset)
        training_epoch_loss.append(train_loss)
        training_epoch_mee.append(train_mee)


        # Validation
        model.eval()
        val_loss = 0.0
        distances = 0.0
        total = 0

        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                outputs = model(inputs)

                val_loss += criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                distances+=torch.sum(torch.sqrt(torch.sum((outputs - labels)**2, dim=1))).item()

        # Calculate the validation loss and mee
        val_mee = distances / total
        val_loss /= len(val_dataloader.dataset)
        validation_epoch_mee.append(val_mee)
        validation_epoch_loss.append(val_loss)


    # Plot mean training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_epoch_mee, label='Train')
    plt.plot(validation_epoch_mee, label='Internal Test')
    plt.xlabel('Epoch')
    plt.ylabel('MEE')
    plt.legend()

    # Plot mean training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(training_epoch_loss, label='Train')
    plt.plot(validation_epoch_loss, label='Internal Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'MEE_internal_test: {validation_epoch_mee[-1]}')
    print(f'MEE_training: {training_epoch_mee[-1]}')
    print(f'MSE_internal_test: {validation_epoch_loss[-1]}')
    print(f'MSE_training: {training_epoch_loss[-1]}')


    torch.save(model.state_dict(), 'CUP_MODEL')
    return model









##KNN




def grid_search_knn_regressor(param_grid, X_train, y_train):

    knn_regressor = KNeighborsRegressor()




    grid= GridSearchCV(
        estimator=knn_regressor, 
        param_grid=param_grid, 
        cv=KFold(n_splits = 5),
        n_jobs=-1)


    grid.fit(X_train, y_train)


    print(grid.best_params_)
    return grid.best_params_, grid.best_estimator_



def best_knn_regressor_validation(best_estimator, X_train, Y_train):
    n_splits = 5
    mee_list = []
    classification_accuracy_list = []
    train_mee_list = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

        #split the data
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        best_estimator.fit(x_train, y_train)
        train_pred = best_estimator.predict(x_train)
        val_pred = best_estimator.predict(x_val)

        train_mee =  mean_euclidean_error(y_true = y_train, y_pred = train_pred)
        val_mee =mean_euclidean_error(y_true = y_val, y_pred = val_pred)

        print(f"Train MEE: {train_mee}")
        print(f"Validation MEE: {val_mee}")

        mee_list.append(val_mee)
        train_mee_list.append(train_mee)


    #compute average MEE
    avg_mee = np.mean(mee_list)
    avg_train_mee = np.mean(train_mee_list)


    print(f"\nAverage Training MEE Across Folds: {avg_train_mee}")
    print(f"Average Validation MEE Across Folds: {avg_mee}")



def best_knn_regressor_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)


    train_mee =  mean_euclidean_error(y_true = y_train, y_pred = train_pred)
    test_mee =mean_euclidean_error(y_true = y_test, y_pred = test_pred)

    print(f"Train MEE: {train_mee}")
    print(f"Test MEE: {test_mee}")




# example of use scikit_randomized_cup(X_train, y_train, param_grid, SVR())
def scikit_randomized_cup(X_train, y_train, param, model):
    np.random.seed(0)
    
    def mean_euclidean_error(y_true, y_pred):
        return np.mean(np.sqrt(np.sum((y_true - y_pred)**2, axis=1)))

    #create a custom scorer
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    
    RS= RandomizedSearchCV(
    MultiOutputRegressor(model),
    param_distributions=param,
    cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=1),
    scoring=mee_scorer,
    n_iter=50,
    verbose = 0,
    n_jobs=-1)


    #fit the models and compute mean accuracy and standard deviation
    RS.fit(X_train, y_train)
    
    results = RS.cv_results_
    
    #get the indices sorted by the mean test score
    sorted_indices = np.argsort(results['mean_test_mee'])
    top_two_indices = sorted_indices[-2:]

    #get the corresponding parameter settings
    top_two_parameters = [results['params'][i] for i in top_two_indices]
    
    
    
    #FOR THE BEST MODEL ONLY WE PLOT LOSS AND  ACCURACY
    clf=RS.best_estimator_
    
    #compute cross-validated accuracy scores
    cv_scores = cross_val_score(clf, X_train, y_train, scoring=mee_scorer, cv=RepeatedKFold(n_splits=7, n_repeats=5, random_state=1))
    mean = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    
    
    print(f'Mean_mee: {mean}, Standard_deviation: {std_dev}')
    print(top_two_parameters)
    
    
    



def scikit_grid_cup(X_train, y_train, param, model):
    np.random.seed(0)
    def mean_euclidean_error(y_true, y_pred):
        return np.mean(np.sqrt(np.sum((y_true - y_pred)**2, axis=1)))

    #create a custom scorer
    mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)
    
    param_grid=param
    
    grid= GridSearchCV(
    MultiOutputRegressor(model),
    param_grid=param_grid,
    cv=RepeatedKFold(n_splits=7, n_repeats=5, random_state=1),
    scoring=mee_scorer,
    verbose = 0,
    n_jobs=-1)
    
    #fit the models and compute mean accuracy and standard deviation
    grid.fit(X_train, y_train)
    
    #save the best model and parameters
    parameters=grid.best_params_
    clf=grid.best_estimator_
    
    #compute cross-validated accuracy scores
    cv_scores = cross_val_score(clf, X_train, y_train, scoring=mee_scorer, cv=RepeatedKFold(n_splits=7, n_repeats=5, random_state=1))
    mean = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    
    #to get the training loss and validation loss
    clf.fit(X_train, y_train)
    train_loss=clf.loss_curve_
    
    #create a figure
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    
    print(parameters)
    return clf



def grid_search_xgboost_regressor(param_grid, X_train, y_train):
    
    xgboost_regressor = XGBRegressor()



    
    grid= GridSearchCV(
        estimator=xgboost_regressor, 
        param_grid=param_grid, 
        cv=KFold(n_splits = 5),
        n_jobs=-1)

    
    grid.fit(X_train, y_train)

   
    print(grid.best_params_)
    return grid.best_params_, grid.best_estimator_



def best_xgboost_regressor_validation(best_estimator, X_train, Y_train):
    n_splits = 5
    mee_list = []
    classification_accuracy_list = []
    train_mee_list = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

    
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        best_estimator.fit(x_train, y_train)
        train_pred = best_estimator.predict(x_train)
        val_pred = best_estimator.predict(x_val)

        train_mee =  mean_euclidean_error(y_true = y_train, y_pred = train_pred)
        val_mee =mean_euclidean_error(y_true = y_val, y_pred = val_pred)

        print(f"Train MEE: {train_mee}")
        print(f"Validation MEE: {val_mee}")

        mee_list.append(val_mee)
        train_mee_list.append(train_mee)


   
    avg_mee = np.mean(mee_list)
    avg_train_mee = np.mean(train_mee_list)


    print(f"\nAverage Training MEE Across Folds: {avg_train_mee}")
    print(f"Average Validation MEE Across Folds: {avg_mee}")


def best_xgboost_regressor_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)


    train_mee =  mean_euclidean_error(y_true = y_train, y_pred = train_pred)
    test_mee =mean_euclidean_error(y_true = y_test, y_pred = test_pred)

    print(f"Train MEE: {train_mee}")
    print(f"Test MEE: {test_mee}")


def grid_search_svr(param_grid, X_train, y_train):

    svr = SVR()

    multi_output_svr = MultiOutputRegressor(svr)

 
    grid= GridSearchCV(
        estimator=multi_output_svr, 
        param_grid=param_grid, 
        cv=KFold(n_splits = 5),
        n_jobs=-1)


    

    grid.fit(X_train, y_train)

   
    print(grid.best_params_)
    return grid.best_params_, grid.best_estimator_








def best_svr_validation(best_estimator, X_train, Y_train):
    n_splits = 5
    mee_list = []
    classification_accuracy_list = []
    train_mee_list = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

   
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        best_estimator.fit(x_train, y_train)
        train_pred = best_estimator.predict(x_train)
        val_pred = best_estimator.predict(x_val)

        train_mee =  mean_euclidean_error(y_true = y_train, y_pred = train_pred)
        val_mee =mean_euclidean_error(y_true = y_val, y_pred = val_pred)

        print(f"Train MEE: {train_mee}")
        print(f"Validation MEE: {val_mee}")

        mee_list.append(val_mee)
        train_mee_list.append(train_mee)


 
    avg_mee = np.mean(mee_list)
    avg_train_mee = np.mean(train_mee_list)


    print(f"\nAverage Training MEE Across Folds: {avg_train_mee}")
    print(f"Average Validation MEE Across Folds: {avg_mee}")


def best_svr_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)


    train_mee =  mean_euclidean_error(y_true = y_train, y_pred = train_pred)
    test_mee =mean_euclidean_error(y_true = y_test, y_pred = test_pred)

    print(f"Train MEE: {train_mee}")
    print(f"Test MEE: {test_mee}")