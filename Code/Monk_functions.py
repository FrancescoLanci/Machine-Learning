import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.svm import SVR
import tensorflow as tf

import torch
from torch import nn
from torch.optim import SGD
from skorch import NeuralNetClassifier
from torch.nn import MSELoss
import torch.nn.functional as F


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras import regularizers
from sklearn.neighbors import KNeighborsRegressor


import random
import itertools



from scipy.stats import uniform, randint, loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, mean_squared_error, classification_report, confusion_matrix


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader






# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)



#preprocssing of the monk
def monk_transform(monk_number):
    monk=pd.read_csv(monk_number, sep=' ', header=None, skipinitialspace=True)
    
    #here we drop the last column which contains the id of the record (not useful)
    monk=monk.iloc[:,:-1]

    #one hot encoding --- #we get one colum for each categorical attribute class except the first column (target variable)
    monk = pd.get_dummies(monk, columns=monk.columns[1:])

    #we split the target variable from the other data
    y=monk[0]
    X=monk.iloc[:,1:]
    
    
    #data transformed in numpy arrays for efficiency
    y=np.array(y)
    X=X.values
    
    return X, y



def scikit_randomized(X_train, y_train, X_test, y_test, param):
    np.random.seed(0)
    # custom scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    RS= RandomizedSearchCV(
    MLPClassifier(random_state=123),
    param_distributions=param,
    cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1),
    scoring=mse_scorer,
    n_iter=100,
    verbose = 0,
    n_jobs=-1)


    #fit the models and compute mean accuracy and standard deviation
    RS.fit(X_train, y_train)
    
    results = RS.cv_results_
    
    #get the indices sorted by the mean test score
    sorted_indices = np.argsort(results['mean_test_score'])
    top_two_indices = sorted_indices[-2:]

    #get the corresponding parameter settings
    top_two_parameters = [results['params'][i] for i in top_two_indices]
    
    
    
    #FOR THE BEST MODEL ONLY WE PLOT LOSS AND  ACCURACY
    clf=RS.best_estimator_
    
    #compute cross-validated accuracy scores
    cv_scores = cross_val_score(clf, X_train, y_train, cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1))
    mean = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    
    
    y_hat = clf.predict(X_test)
    print(f'Mean_accuracy: {mean}, Standard_deviation: {std_dev}')
    print(classification_report(y_test, y_hat))
    print(top_two_parameters)




def scikit_grid(X_train, y_train, X_test, y_test, param):
    np.random.seed(0)
    # Define your custom scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    param_grid=param
    
    grid= GridSearchCV(
    MLPClassifier(random_state=123),
    param_grid=param_grid,
    cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1),
    scoring=mse_scorer,
    verbose = 0,
    n_jobs=-1)
    
    #fit the models and compute mean accuracy and standard deviation
    grid.fit(X_train, y_train)
    
    #save the best model and parameters
    parameters=grid.best_params_
    clf=grid.best_estimator_
    
    #compute cross-validated accuracy scores
    cv_scores = cross_val_score(clf, X_train, y_train, cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1))
    mean = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    
    #to get the training loss and validation loss
    clf.fit(X_train, y_train)
    train_loss=clf.loss_curve_
    
    # Create a figure
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    
    
    y_hat = clf.predict(X_test)
    print(parameters)
    print(f'Mean_accuracy: {mean}, Standard_deviation: {std_dev}')
    print(classification_report(y_test, y_hat))



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
    

#keras classifier: 1 hidden layer, sigmoid as output
def create_keras_classifier(units, activation, learning_rate, momentum, l2, x_train):
    model = Sequential()
    model.add(Dense(units, input_dim=x_train.shape[1], 
                    kernel_regularizer=regularizers.l2(l2), activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model
    
    
#randomized search for a first look on a few combinations of hyperparameters
#it plots the accuracy and loss for every combination and return the n_best loss and their combinations of hyperparameters
def randomized_search_keras_classifier(X_train, Y_train, param_dict, number_of_combinations, n_best):

    n_splits = 5  #number of splits for stratified k-fold

    random_combinations = random.sample(list(itertools.product(*param_dict.values())), number_of_combinations)

    best_combinations = [None] * n_best #where we store the n_best combinations of hyperparameters
    best_losses = [float('inf')] * n_best #where we store the n_best losses

    for combo in random_combinations:
        param_dict = dict(zip(param_dict.keys(), combo))
        print("Testing with hyperparameters:", param_dict)

        #unpack hyperparameters
        units = param_dict['units']
        activation = param_dict['activation']
        learning_rate = param_dict['learning_rate']
        momentum = param_dict['momentum']
        l2 = param_dict['l2']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']
        
        #build the model
        keras_model = KerasClassifier(model=create_keras_classifier, x_train = X_train, units=units, activation = activation, learning_rate=learning_rate, momentum=momentum, l2=l2, optimizer='sgd', verbose = 0)

        # where we append the stats for each run of the cross validation
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
            print(f"Fold {fold + 1}:")

            #split the data
            x_train, y_train = X_train[train_indices], Y_train[train_indices]
            x_val, y_val = X_train[val_indices], Y_train[val_indices]

 

            #fit the model
            keras_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs = epochs, batch_size = batch_size)
    
            # history
            train_loss, train_accuracy = keras_model.history_['loss'], keras_model.history_['accuracy']
            val_loss, val_accuracy = keras_model.history_['val_loss'], keras_model.history_['val_accuracy']
            
            print(f"Training Accuracy: {train_accuracy[-1]}")
            print(f"Training Loss: {train_loss[-1]}")
            print(f"Validation Accuracy: {val_accuracy[-1]}")
            print(f"Validation Loss: {val_loss[-1]}")

            #append the stats for the this run
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)         
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

        #calculate mean loss and accuracy over all folds
        train_mean_loss = [sum(losses) / len(losses) for losses in zip(*train_losses)]
        train_mean_accuracy = [sum(accuracies) / len(accuracies) for accuracies in zip(*train_accuracies)]
        val_mean_loss = [sum(losses) / len(losses) for losses in zip(*val_losses)]
        val_mean_accuracy = [sum(accuracies) / len(accuracies) for accuracies in zip(*val_accuracies)]

        #take the last validation loss, we will use it for the selection of the best losses
        last_epoch_val_mean_loss = val_mean_loss[-1]

        #update the best losses and best combinations
        insert_in_best(best_losses, best_combinations, last_epoch_val_mean_loss, param_dict)

        #plot mean training and validation accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_mean_accuracy, label='Train')
        plt.plot(val_mean_accuracy, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        #plot standard deviation for training and validation accuracy
        train_accuracy_std = np.std(train_accuracies, axis=0)
        val_accuracy_std = np.std(val_accuracies, axis=0)

        plt.fill_between(range(epochs), np.array(train_mean_accuracy) - train_accuracy_std, np.array(train_mean_accuracy) + train_accuracy_std, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), np.array(val_mean_accuracy) - val_accuracy_std, np.array(val_mean_accuracy) + val_accuracy_std, alpha=0.3, label='Validation Std Dev')

        plt.legend()

        #plot mean training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(train_mean_loss, label='Train')
        plt.plot(val_mean_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        #plot standard deviation for training and validation loss
        train_loss_std = np.std(train_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)

        plt.fill_between(range(epochs), np.array(train_mean_loss) - train_loss_std, np.array(train_mean_loss) + train_loss_std, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), np.array(val_mean_loss) - val_loss_std, np.array(val_mean_loss) + val_loss_std, alpha=0.3, label='Validation Std Dev')

        plt.legend()
        plt.tight_layout()
        plt.show()


    #best parameters and results
    
    for i in range(len(best_losses)):
        print(i, best_losses[i], best_combinations[i])
        


# grid search, returns the best parameters and the best estimator
def grid_search_keras_classifier(param_grid, X_train, y_train):
    model = KerasClassifier(model=create_keras_classifier, optimizer = 'sgd', x_train=X_train)


    #create the grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        verbose=3,
        return_train_score=True
    )


    grid.fit(X_train, y_train, verbose=1)
    print(grid.best_params_)
    return grid.best_params_, grid.best_estimator_


#for the plot of the best_estimator on the validations from the stratified k fold
def plot_best_keras_classifier_validation(best_estimator, epochs, X_train, Y_train):

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

        #split the data
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

    

        #fit the model
        best_estimator.fit(x_train, y_train, validation_data=(x_val, y_val), verbose = 0)

        #history
        train_loss, train_accuracy = best_estimator.history_['loss'], best_estimator.history_['accuracy']
        val_loss, val_accuracy = best_estimator.history_['val_loss'], best_estimator.history_['val_accuracy']

        print(f"Training Accuracy: {train_accuracy[-1]}")
        print(f"Training Loss: {train_loss[-1]}")
        print(f"Validation Accuracy: {val_accuracy[-1]}")
        print(f"Validation Loss: {val_loss[-1]}")

        #append the stats for the this run
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)         
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

    #calculate mean loss and accuracy over all folds
    train_mean_loss = [sum(losses) / len(losses) for losses in zip(*train_losses)]
    train_mean_accuracy = [sum(accuracies) / len(accuracies) for accuracies in zip(*train_accuracies)]
    val_mean_loss = [sum(losses) / len(losses) for losses in zip(*val_losses)]
    val_mean_accuracy = [sum(accuracies) / len(accuracies) for accuracies in zip(*val_accuracies)]

    #take the last validation loss, we will use it for the selection of the best losses
    last_epoch_val_mean_loss = val_mean_loss[-1]



    #plot mean training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_mean_accuracy, label='Train')
    plt.plot(val_mean_accuracy, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    #plot standard deviation for training and validation accuracy
    train_accuracy_std = np.std(train_accuracies, axis=0)
    val_accuracy_std = np.std(val_accuracies, axis=0)

    plt.fill_between(range(epochs), np.array(train_mean_accuracy) - train_accuracy_std, np.array(train_mean_accuracy) + train_accuracy_std, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), np.array(val_mean_accuracy) - val_accuracy_std, np.array(val_mean_accuracy) + val_accuracy_std, alpha=0.3, label='Validation Std Dev')

    plt.legend()

    #plot mean training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_mean_loss, label='Train')
    plt.plot(val_mean_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot standard deviation for training and validation loss
    train_loss_std = np.std(train_losses, axis=0)
    val_loss_std = np.std(val_losses, axis=0)

    plt.fill_between(range(epochs), np.array(train_mean_loss) - train_loss_std, np.array(train_mean_loss) + train_loss_std, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), np.array(val_mean_loss) - val_loss_std, np.array(val_mean_loss) + val_loss_std, alpha=0.3, label='Validation Std Dev')

    plt.legend()
    plt.tight_layout()
    plt.show()
    print(last_epoch_val_mean_loss)


#plots of the best model on the test
def plot_best_keras_classifier_test(X_train, y_train, X_test, y_test, best_estimator):
    
    best_estimator.fit(X_train, y_train, validation_data=(X_test, y_test))
    y_pred = best_estimator.predict(X_test)

    #plot training & validation accuracy values
    plt.plot(best_estimator.history_['accuracy'])
    plt.plot(best_estimator.history_['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    #plot training & validation loss values
    plt.plot(best_estimator.history_['loss'])
    plt.plot(best_estimator.history_['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    print(classification_report(y_test, y_pred))
    print(best_estimator.history_['val_loss'][-1])



class torch_Model(nn.Module):
    def __init__(self, input_size, units, weight_init=torch.nn.init.xavier_uniform_):
        super().__init__()
        self.layer = nn.Linear(input_size, units)
        self.act = nn.Tanh()
        self.output = nn.Linear(units, 1)
        self.prob = nn.Sigmoid()
        # manually init weights
        weight_init(self.layer.weight)
        weight_init(self.output.weight)

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def create_torch_classifier(input_size, units, learning_rate, momentum, l2):
    model = torch_Model(input_size, units)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2)
    criterion = nn.MSELoss()
    return model, optimizer, criterion






def randomized_search_pytorch(X_train, Y_train, param_dict, number_of_combinations, n_best, delta=0.001, seed=0):
    torch.manual_seed(seed)
    Y_train = Y_train.reshape(-1, 1)

    n_splits = 5

    random.seed(0)
    random_combinations = random.sample(list(itertools.product(*param_dict.values())), number_of_combinations)

    best_combinations = [None] * n_best
    best_losses = [float('inf')] * n_best

    for combo in random_combinations:
        param_dict = dict(zip(param_dict.keys(), combo))
        print("Testing with hyperparameters:", param_dict)

        units = param_dict['units']
        learning_rate = param_dict['learning_rate']
        momentum = param_dict['momentum']
        l2 = param_dict['l2']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']

        all_train_accuracies = []
        all_train_losses = []
        all_val_accuracies = []
        all_val_losses = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
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
            
            #build the model
            model, optimizer, criterion = create_torch_classifier(input_size, units, learning_rate, momentum, l2)

            training_epoch_loss = []
            validation_epoch_loss = []
            training_epoch_accuracy = []
            validation_epoch_accuracy = []

            #fit the model
            #early_stopper = EarlyStopper(patience=10, min_delta=delta)
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                running_corrects = 0
                running_total = 0
                
                for i, (inputs, labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    
                    #convert probability outputs to binary predictions
                    predictions = (outputs > 0.5).float()
                    running_corrects += torch.sum(predictions == labels).item()
                    running_total += labels.size(0)

                #calculate the training loss and training accuracy
                train_loss = running_loss / len(train_dataloader.dataset)
                train_accuracy = 100 * running_corrects / running_total
                training_epoch_loss.append(train_loss)
                training_epoch_accuracy.append(train_accuracy)

                #validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for data in val_dataloader:
                        inputs, labels = data
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item() * labels.size(0)

                        predictions = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += torch.sum(predictions == labels).item()

                #calculate the validation accuracy and validation loss
                val_accuracy = 100 * correct / total
                val_loss /= len(val_dataloader.dataset)
                validation_epoch_accuracy.append(val_accuracy)
                validation_epoch_loss.append(val_loss)
                #if early_stopper.early_stop(val_loss):             
                    #break

                
            print(f"Training Accuracy: {training_epoch_accuracy[-1]}")
            print(f"Training Loss: {training_epoch_loss[-1]}")
            print(f"Validation Accuracy: {validation_epoch_accuracy[-1]}")
            print(f"Validation Loss: {validation_epoch_loss[-1]}") 
            #store results for this fold
            
            
            all_train_accuracies.append(training_epoch_accuracy)
            all_train_losses.append(training_epoch_loss)
            all_val_accuracies.append(validation_epoch_accuracy)
            all_val_losses.append(validation_epoch_loss)
            
           

        #calculate mean and standard deviation across folds
        mean_train_accuracy=np.mean(all_train_accuracies, axis=0)
        mean_train_loss = np.mean(all_train_losses, axis=0)
        std_train_accuracy = np.std(all_train_accuracies, axis=0)
        std_train_loss = np.std(all_train_losses, axis=0)
        
        mean_val_accuracy = np.mean(all_val_accuracies, axis=0)
        mean_val_loss = np.mean(all_val_losses, axis=0)
        std_val_accuracy = np.std(all_val_accuracies, axis=0)
        std_val_loss = np.std(all_val_losses, axis=0)

        #plot mean training and validation accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(mean_train_accuracy, label='Train')
        plt.plot(mean_val_accuracy, label='Validation')
        plt.fill_between(range(epochs), mean_train_accuracy - std_train_accuracy, mean_train_accuracy + std_train_accuracy, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_accuracy - std_val_accuracy, mean_val_accuracy + std_val_accuracy, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        #plot mean training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(mean_train_loss, label='Train')
        plt.plot(mean_val_loss, label='Validation')
        plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        #insert the best combination and loss into the list
        insert_in_best(best_losses, best_combinations, mean_val_loss[-1], param_dict)

    #print the best combinations and their losses
    print("\nBest Combinations:")
    for i in range(n_best):
        print(f"{i+1}. Hyperparameters: {best_combinations[i]}, Mean Validation Loss: {best_losses[i]:.4f}")




def grid_search_pytorch(X_train, Y_train, param_dict, n_best, seed=0):
    torch.manual_seed(seed)
    Y_train = Y_train.reshape(-1, 1)

    n_splits = 5

    combinations = list(itertools.product(*param_dict.values()))
    
    best_combinations = [None] * n_best
    best_losses = [float('inf')] * n_best

    for combo in combinations:
        param_dict = dict(zip(param_dict.keys(), combo))
        print("Testing with hyperparameters:", param_dict)

        units = param_dict['units']
        learning_rate = param_dict['learning_rate']
        momentum = param_dict['momentum']
        l2 = param_dict['l2']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']

        all_train_accuracies = []
        all_train_losses = []
        all_val_accuracies = []
        all_val_losses = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
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
            
            #build the model
            model, optimizer, criterion = create_torch_classifier(input_size, units, learning_rate, momentum, l2)

            training_epoch_loss = []
            validation_epoch_loss = []
            training_epoch_accuracy = []
            validation_epoch_accuracy = []

            #fit the model
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                running_corrects = 0
                running_total = 0
                
                for i, (inputs, labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    
                    #convert probability outputs to binary predictions
                    predictions = (outputs > 0.5).float()
                    running_corrects += torch.sum(predictions == labels).item()
                    running_total += labels.size(0)

                #calculate the training loss and training accuracy
                train_loss = running_loss / len(train_dataloader.dataset)
                train_accuracy = 100 * running_corrects / running_total
                training_epoch_loss.append(train_loss)
                training_epoch_accuracy.append(train_accuracy)

                #validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item() * labels.size(0)

                        predictions = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += torch.sum(predictions == labels).item()

                #calculate the validation accuracy and validation loss
                val_accuracy = 100 * correct / total
                val_loss /= len(val_dataloader.dataset)
                validation_epoch_accuracy.append(val_accuracy)
                validation_epoch_loss.append(val_loss)

                
            print(f"Training Accuracy: {training_epoch_accuracy[-1]}")
            print(f"Training Loss: {training_epoch_loss[-1]}")
            print(f"Validation Accuracy: {validation_epoch_accuracy[-1]}")
            print(f"Validation Loss: {validation_epoch_loss[-1]}") 
            #store results for this fold
            
            
            all_train_accuracies.append(training_epoch_accuracy)
            all_train_losses.append(training_epoch_loss)
            all_val_accuracies.append(validation_epoch_accuracy)
            all_val_losses.append(validation_epoch_loss)
            
           

        #calculate mean and standard deviation across folds
        mean_train_accuracy = np.mean(all_train_accuracies, axis=0)
        mean_train_loss = np.mean(all_train_losses, axis=0)
        std_train_accuracy = np.std(all_train_accuracies, axis=0)
        std_train_loss = np.std(all_train_losses, axis=0)

        mean_val_accuracy = np.mean(all_val_accuracies, axis=0)
        mean_val_loss = np.mean(all_val_losses, axis=0)
        std_val_accuracy = np.std(all_val_accuracies, axis=0)
        std_val_loss = np.std(all_val_losses, axis=0)

        #plot mean training and validation accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(mean_train_accuracy, label='Train')
        plt.plot(mean_val_accuracy, label='Validation')
        plt.fill_between(range(epochs), mean_train_accuracy - std_train_accuracy, mean_train_accuracy + std_train_accuracy, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_accuracy - std_val_accuracy, mean_val_accuracy + std_val_accuracy, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        #plot mean training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(mean_train_loss, label='Train')
        plt.plot(mean_val_loss, label='Validation')
        plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, label='Train Std Dev')
        plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, label='Validation Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        #insert the best combination and loss into the list
        insert_in_best(best_losses, best_combinations, mean_val_loss[-1], param_dict)

    #print the best combinations and their losses
    print("\nBest Combinations:")
    to_return=[]
    for i in range(n_best):
        print(f"{i+1}. Hyperparameters: {best_combinations[i]}, Mean Validation Loss: {best_losses[i]:.4f}")
        to_return.append(best_combinations[i])

    return to_return



def plot_best_torch_classifier_validation(best_estimator, X_train, Y_train, n_splits, seed=0):
    torch.manual_seed(seed)
    Y_train = Y_train.reshape(-1, 1)
    
    all_train_accuracies = []
    all_train_losses = []
    all_val_accuracies = []
    all_val_losses = []
    
    units = best_estimator['units']
    learning_rate = best_estimator['learning_rate']
    momentum = best_estimator['momentum']
    l2 = best_estimator['l2']
    batch_size = best_estimator['batch_size']
    epochs = best_estimator['epochs']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):

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

        #build the model
        model, optimizer, criterion = create_torch_classifier(input_size, units, learning_rate, momentum, l2)

        training_epoch_loss = []
        validation_epoch_loss = []
        training_epoch_accuracy = []
        validation_epoch_accuracy = []

        #fit the model
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                #convert probability outputs to binary predictions
                predictions = (outputs > 0.5).float()
                running_corrects += torch.sum(predictions == labels).item()
                running_total += labels.size(0)

            #calculate the training loss and training accuracy
            train_loss = running_loss / len(train_dataloader.dataset)
            train_accuracy = 100 * running_corrects / running_total
            training_epoch_loss.append(train_loss)
            training_epoch_accuracy.append(train_accuracy)

            #validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data in val_dataloader:
                    images, labels = data
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item() * labels.size(0)

                    predictions = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += torch.sum(predictions == labels).item()

            #calculate the validation accuracy and validation loss
            val_accuracy = 100 * correct / total
            val_loss /= len(val_dataloader.dataset)
            validation_epoch_accuracy.append(val_accuracy)
            validation_epoch_loss.append(val_loss)


        all_train_accuracies.append(training_epoch_accuracy)
        all_train_losses.append(training_epoch_loss)
        all_val_accuracies.append(validation_epoch_accuracy)
        all_val_losses.append(validation_epoch_loss)



    #calculate mean and standard deviation across folds
    mean_train_accuracy = np.mean(all_train_accuracies, axis=0)
    mean_train_loss = np.mean(all_train_losses, axis=0)
    std_train_accuracy = np.std(all_train_accuracies, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)

    mean_val_accuracy = np.mean(all_val_accuracies, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    std_val_accuracy = np.std(all_val_accuracies, axis=0)
    std_val_loss = np.std(all_val_losses, axis=0)

    #plot mean training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mean_train_accuracy, label='Train')
    plt.plot(mean_val_accuracy, label='Validation')
    plt.fill_between(range(epochs), mean_train_accuracy - std_train_accuracy, mean_train_accuracy + std_train_accuracy, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), mean_val_accuracy - std_val_accuracy, mean_val_accuracy + std_val_accuracy, alpha=0.3, label='Validation Std Dev')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #plot mean training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(mean_train_loss, label='Train')
    plt.plot(mean_val_loss, label='Validation')
    plt.fill_between(range(epochs), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3, label='Train Std Dev')
    plt.fill_between(range(epochs), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3, label='Validation Std Dev')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()




def retraining_pytorch(final_parameters, X, y, X_val, y_val, seed=0):
    torch.manual_seed(seed)
    def torch_format(X, y):
        X=torch.FloatTensor(X)
        y=torch.FloatTensor(y).reshape(-1,1)
        return X, y

    X_train, y_train=torch_format(X, y)
    X_test, y_test=torch_format(X_val, y_val)
    input_size = X_train.shape[1]
    
    
    units = final_parameters['units']
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
    model, optimizer, criterion = create_torch_classifier(input_size, units, learning_rate, momentum, l2)
    training_epoch_loss = []
    validation_epoch_loss = []
    training_epoch_accuracy = []
    validation_epoch_accuracy = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            #convert probability outputs to binary predictions
            predictions = (outputs > 0.5).float()
            running_corrects += torch.sum(predictions == labels).item()
            running_total += labels.size(0)

        #calculate the training loss and training accuracy
        train_loss = running_loss / len(train_dataloader.dataset)
        train_accuracy = 100 * running_corrects / running_total
        training_epoch_loss.append(train_loss)
        training_epoch_accuracy.append(train_accuracy)

        #validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        
        CM=0
        with torch.no_grad():
            for data in val_dataloader:
                images, labels = data
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * labels.size(0)

                predictions = (outputs > 0.5).float()
                total += labels.size(0)
                correct += torch.sum(predictions == labels).item()
                
                CM+=confusion_matrix(labels, predictions, labels=[0,1])

        #calculate the validation accuracy and validation loss
        val_accuracy = 100 * correct / total
        val_loss /= len(val_dataloader.dataset)
        validation_epoch_accuracy.append(val_accuracy)
        validation_epoch_loss.append(val_loss)


    #plot mean training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_epoch_accuracy, label='Train')
    plt.plot(validation_epoch_accuracy, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #plot mean training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(training_epoch_loss, label='Train')
    plt.plot(validation_epoch_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    sensitivity=tp/(tp+fn)
    precision=tp/(tp+fp)

    print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
    print()
    print('Confusion Matirx : ')
    print(CM)
    print('- Sensitivity : ',(tp/(tp+fn))*100)
    print('- Specificity : ',(tn/(tn+fp))*100)
    print('- Precision: ',(tp/(tp+fp))*100)
    print('- NPV: ',(tn/(tn+fn))*100)
    print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)

    print(f'Test MSE: {validation_epoch_loss[-1]}')
    print(f'train MSE: {training_epoch_loss[-1]}')
    print()




def grid_search_knn_classifier(param_grid, X_train, y_train):
    #create a KNeighborsRegressor
    knn_regressor = KNeighborsRegressor()




    grid= GridSearchCV(
        estimator=knn_regressor, 
        param_grid=param_grid, 
        cv=StratifiedKFold(n_splits = 5),
        n_jobs=-1)


    grid.fit(X_train, y_train)


    print(grid.best_params_)
    return grid.best_params_, grid.best_estimator_


def best_knn_classifier_validation(best_estimator, X_train, Y_train):
    n_splits = 5
    mse_list = []
    classification_accuracy_list = []
    train_mse_list = []
    train_classification_accuracy_list = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

        # split the data
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        best_estimator.fit(x_train, y_train)
        train_pred = best_estimator.predict(x_train)
        val_pred = best_estimator.predict(x_val)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)

        print(f"Train MSE: {train_mse}")
        print(f"Validation MSE: {val_mse}")

        mse_list.append(val_mse)
        train_mse_list.append(train_mse)

        #classification (rounding to 0 or 1)
        train_pred_class = np.round(train_pred)
        val_pred_class = np.round(val_pred)

        #compute classification accuracy
        train_accuracy = np.mean(train_pred_class == y_train)
        val_accuracy = np.mean(val_pred_class == y_val)

        print(f"Train Classification Accuracy: {train_accuracy}")
        print(f"Validation Classification Accuracy: {val_accuracy}")

        classification_accuracy_list.append(val_accuracy)
        train_classification_accuracy_list.append(train_accuracy)

    #compute average MSE and classification accuracy across folds
    avg_mse = np.mean(mse_list)
    avg_classification_accuracy = np.mean(classification_accuracy_list)
    avg_train_mse = np.mean(train_mse_list)
    avg_train_classification_accuracy = np.mean(train_classification_accuracy_list)

    print(f"\nAverage Training MSE Across Folds: {avg_train_mse}")
    print(f"Average Training Classification Accuracy Across Folds: {avg_train_classification_accuracy}")
    print(f"Average Validation MSE Across Folds: {avg_mse}")
    print(f"Average Validation Classification Accuracy Across Folds: {avg_classification_accuracy}")





def best_knn_classifier_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    
    train_pred_class = np.round(train_pred)
    test_pred_class = np.round(test_pred)

    #compute classification accuracy
    train_accuracy = np.mean(train_pred_class == y_train)
    test_accuracy = np.mean(test_pred_class == y_test)

    print(f"Train Classification Accuracy: {train_accuracy}")
    print(f"Test Classification Accuracy: {test_accuracy}")
       





def scikit_randomized_monk(X_train, y_train, X_test, y_test, param, model):
    np.random.seed(0)
    #custom scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    RS= RandomizedSearchCV(
    model,
    param_distributions=param,
    cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1),
    scoring=mse_scorer,
    n_iter=100,
    verbose = 0,
    n_jobs=-1)


    #fit the models and compute mean accuracy and standard deviation
    RS.fit(X_train, y_train)
    
    results = RS.cv_results_
    
    #get the indices sorted by the mean test score
    sorted_indices = np.argsort(results['mean_test_score'])
    top_two_indices = sorted_indices[-2:]

    #get the corresponding parameter settings
    top_two_parameters = [results['params'][i] for i in top_two_indices]
    
    
    
    #FOR THE BEST MODEL ONLY WE PLOT LOSS AND  ACCURACY
    clf=RS.best_estimator_
    
    #compute cross-validated accuracy scores
    cv_scores = cross_val_score(clf, X_train, y_train, scoring=mse_scorer, cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1))
    mean = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    
    
    y_hat = clf.predict(X_test)
    print(f'Mean_accuracy: {mean}, Standard_deviation: {std_dev}')
    print(classification_report(y_test, y_hat))
    print(top_two_parameters)
    
    




def scikit_grid_monk(X_train, y_train, X_test, y_test, param, model):
    np.random.seed(0)
    #ustom scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    param_grid=param
    
    grid= GridSearchCV(
    model,
    param_grid=param_grid,
    cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1),
    scoring=mse_scorer,
    verbose = 1,
    n_jobs=-1)
    
    #fit the models and compute mean accuracy and standard deviation
    grid.fit(X_train, y_train)
    
    #save the best model and parameters
    parameters=grid.best_params_
    clf=grid.best_estimator_
    
    #compute cross-validated accuracy scores
    cv_scores = cross_val_score(clf, X_train, y_train, scoring=mse_scorer, cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1))
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
    
    
    
    y_hat = clf.predict(X_test)
    print(parameters)
    print(f'Mean_accuracy: {mean}, Standard_deviation: {std_dev}')
    print(classification_report(y_test, y_hat))
    return clf
    
    
def best_MLP_classifier_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict_proba(X_train)[:, 1]
    test_pred = best_estimator.predict_proba(X_test)[:, 1]

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    train_pred_class = np.round(train_pred)
    test_pred_class = np.round(test_pred)

    #compute classification accuracy
    train_accuracy = np.mean(train_pred_class == y_train)
    test_accuracy = np.mean(test_pred_class == y_test)

    print(f"Train Classification Accuracy: {train_accuracy}")
    print(f"Test Classification Accuracy: {test_accuracy}")







def grid_search_xgboost_classifier(param_grid, X_train, y_train):
 
    xgboost_regressor = XGBRegressor()





    grid= GridSearchCV(
        estimator=xgboost_regressor, 
        param_grid=param_grid, 
        cv=StratifiedKFold(n_splits = 5),
        n_jobs=-1)


    grid.fit(X_train, y_train)

    print(grid.best_params_, grid.best_score_)
    return grid.best_params_, grid.best_estimator_




def best_xgboost_classifier_validation(best_estimator, X_train, Y_train):
    n_splits = 5
    mse_list = []
    classification_accuracy_list = []
    train_mse_list = []
    train_classification_accuracy_list = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

        #split the data
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        best_estimator.fit(x_train, y_train)
        train_pred = best_estimator.predict(x_train)
        val_pred = best_estimator.predict(x_val)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)

        print(f"Train MSE: {train_mse}")
        print(f"Validation MSE: {val_mse}")

        mse_list.append(val_mse)
        train_mse_list.append(train_mse)

        #classification (rounding to 0 or 1)
        train_pred_class = np.round(train_pred)
        val_pred_class = np.round(val_pred)

        #compute classification accuracy
        train_accuracy = np.mean(train_pred_class == y_train)
        val_accuracy = np.mean(val_pred_class == y_val)

        print(f"Train Classification Accuracy: {train_accuracy}")
        print(f"Validation Classification Accuracy: {val_accuracy}")

        classification_accuracy_list.append(val_accuracy)
        train_classification_accuracy_list.append(train_accuracy)

    #compute average MSE and classification accuracy across folds
    avg_mse = np.mean(mse_list)
    avg_classification_accuracy = np.mean(classification_accuracy_list)
    avg_train_mse = np.mean(train_mse_list)
    avg_train_classification_accuracy = np.mean(train_classification_accuracy_list)

    print(f"\nAverage Training MSE Across Folds: {avg_train_mse}")
    print(f"Average Training Classification Accuracy Across Folds: {avg_train_classification_accuracy}")
    print(f"Average Validation MSE Across Folds: {avg_mse}")
    print(f"Average Validation Classification Accuracy Across Folds: {avg_classification_accuracy}")


def best_xgboost_classifier_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    
    train_pred_class = np.round(train_pred)
    test_pred_class = np.round(test_pred)

    #compute classification accuracy
    train_accuracy = np.mean(train_pred_class == y_train)
    test_accuracy = np.mean(test_pred_class == y_test)

    print(f"Train Classification Accuracy: {train_accuracy}")
    print(f"Test Classification Accuracy: {test_accuracy}")




def grid_search_svc(param_grid, X_train, y_train):

    svc = SVR()





    grid= GridSearchCV(
        estimator=svc, 
        param_grid=param_grid, 
        cv=StratifiedKFold(n_splits = 5),
        n_jobs=-1)


    grid.fit(X_train, y_train)

    print(grid.best_params_, grid.best_score_)
    return grid.best_params_, grid.best_estimator_



def best_svc_validation(best_estimator, X_train, Y_train):
    n_splits = 5
    mse_list = []
    classification_accuracy_list = []
    train_mse_list = []
    train_classification_accuracy_list = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
        print(f"Fold {fold + 1}:")

        #split the data
        x_train, y_train = X_train[train_indices], Y_train[train_indices]
        x_val, y_val = X_train[val_indices], Y_train[val_indices]

        best_estimator.fit(x_train, y_train)
        train_pred = best_estimator.predict(x_train)
        val_pred = best_estimator.predict(x_val)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)

        print(f"Train MSE: {train_mse}")
        print(f"Validation MSE: {val_mse}")

        mse_list.append(val_mse)
        train_mse_list.append(train_mse)

        #classification (rounding to 0 or 1)
        train_pred_class = np.round(train_pred)
        val_pred_class = np.round(val_pred)

        #compute classification accuracy
        train_accuracy = np.mean(train_pred_class == y_train)
        val_accuracy = np.mean(val_pred_class == y_val)

        print(f"Train Classification Accuracy: {train_accuracy}")
        print(f"Validation Classification Accuracy: {val_accuracy}")

        classification_accuracy_list.append(val_accuracy)
        train_classification_accuracy_list.append(train_accuracy)

    #compute average MSE and classification accuracy across folds
    avg_mse = np.mean(mse_list)
    avg_classification_accuracy = np.mean(classification_accuracy_list)
    avg_train_mse = np.mean(train_mse_list)
    avg_train_classification_accuracy = np.mean(train_classification_accuracy_list)

    print(f"\nAverage Training MSE Across Folds: {avg_train_mse}")
    print(f"Average Training Classification Accuracy Across Folds: {avg_train_classification_accuracy}")
    print(f"Average Validation MSE Across Folds: {avg_mse}")
    print(f"Average Validation Classification Accuracy Across Folds: {avg_classification_accuracy}")


def best_svc_test(X_train, y_train, X_test, y_test, best_estimator):

    best_estimator.fit(X_train, y_train)
    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    
    train_pred_class = np.round(train_pred)
    test_pred_class = np.round(test_pred)

    #compute classification accuracy
    train_accuracy = np.mean(train_pred_class == y_train)
    test_accuracy = np.mean(test_pred_class == y_test)

    print(f"Train Classification Accuracy: {train_accuracy}")
    print(f"Test Classification Accuracy: {test_accuracy}")
       