from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import json
import os

class TrainSplit:
    
    ROWID = ['f_0']
    DATE = ['f_1']
    CATEGORIES = [ f'f_{i}' for i in range(2,33) ]
    BINARY = [ f'f_{i}' for i in range(33,42) ]
    NUMERICAL = [ f'f_{i}' for i in range(42,80) ]
    IS_CLICKED = ['is_clicked']
    IS_INSTALLED =['is_installed']

    @staticmethod
    def load_config(name):
        """
        name: String (Name of the model)
        """
        file = open(f'../Data/configs/{name}','rb')
        config = json.load(file)
        file.close()
        return config
    
    @staticmethod
    def get_config_list():
        """
        Return the list of all the config files
        """
        return os.listdir('../Data/configs')

    def __init__(self, val_type='time', class_type='binary',split_date=66,impute=True):

        print("Loading the data")
        self.data = pd.read_csv('../Data/miss_combine.csv')
        self.impute = impute
        self.val_type = val_type
        self.class_type = class_type
        self.split_date = split_date
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.impute_data()
        self.train_test_split()

    def impute_data(self):
        if self.impute:
            self.data['f_30'].fillna(self.data['f_30'].mode()[0],inplace=True)
            self.data['f_31'].fillna(self.data['f_31'].mode()[0],inplace=True)
            print("Categorial Feature Imputed")
            fmiss = "f_43,f_51,f_58,f_59,f_64,f_65,f_66,f_67,f_68,f_69,f_70".split(',')
            for f in tqdm(fmiss,desc="NUM IMPUTE"):
                self.data[f].fillna(self.data[f].mean(),inplace=True)

    def get_split(self):
        """ return self.X_train, self.X_test, self.y_train, self.y_test  """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_split(self):
        print(f"Spliting the Data based on {self.val_type}")
        if self.val_type == 'random':
            self.random_split()
        elif self.val_type == 'time':
            self.time_split()
        elif self.val_type == 'No':
            self.final_split()
        else:
            raise Exception('Invalid validation type')

    def random_split(self):
        """
        Randomly split the data into train and test set
        """
        y = self.data[TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED]
        X = self.data.drop(TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def time_split(self):
        """
        Split the data into train and test set based on Date
        """
        self.X_train = self.data[self.data[TrainSplit.DATE[0]] < self.split_date ].drop(TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED, axis=1)
        self.X_test = self.data[self.data[TrainSplit.DATE[0]] >= self.split_date ].drop(TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED, axis=1)
        self.y_train = self.data[self.data[TrainSplit.DATE[0]] < self.split_date ][TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED]
        self.y_test = self.data[self.data[TrainSplit.DATE[0]] >= self.split_date ][TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED]
        print(f"X_train:{self.X_train.shape}, X_test:{self.X_test.shape} , y_train:{self.y_train.shape} , y_test:{self.y_test.shape}")

    def final_split(self):
        self.X_train = self.data.drop(TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED, axis=1).values
        self.y_train = self.get_label(self.data[TrainSplit.IS_CLICKED + TrainSplit.IS_INSTALLED].values)
        self.X_test = None
        self.y_test = None
    


class TestResults:

    def __init__(self,row_id,is_click,is_install,model_name,config):
        """
        row_id: Pandas series (row_id from the test data)
        is_click: Numpy array (is_click prediction probablity values)
        is_install: Numpy array (is_install prediction probablity values)
        name: String (Meaningful name of the model)
        config: Dict (Hyperparameters of the model)
        This will take the prediction and row_id from the model on test data and save it to csv file
        """
        self.row_id = row_id
        self.is_click = is_click
        self.is_install = is_install
        self.file_name = model_name
        self.config = config
        self.save()

    def save(self):
        """
        Save the prediction to csv file and model config to json file
        """
        result = np.vstack([self.row_id.to_numpy(dtype=int),self.is_click,self.is_install]).T
        df = pd.DataFrame(result,columns=['RowId','is_clicked','is_installed'])
        now = datetime.now()
        df.to_csv(f'../Data/results/{self.file_name}_{now}.csv',index=False,sep='\t')
        print(f"Saved the test result to csv file as {self.file_name}_{now}.csv")
        file = open(f'../Data/configs/{self.file_name}_{now}.json','w')
        json.dump(self.config,file)
        file.close()
        print(f"Saved the model config to json file as {self.file_name}_{now}.json")

        
        
