import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import optuna


class Train_Split:
    
    ROWID = ['f_0']
    DATE = ['f_1']
    CATEGORIES = [ f'f_{i}' for i in range(2,33) ]
    BINARY = [ f'f_{i}' for i in range(33,42) ]
    NUMERICAL = [ f'f_{i}' for i in range(42,80) ]
    IS_CLICKED = ['is_clicked']
    IS_INSTALLED =['is_installed']

    def __init__(self, val_type='random', class_type='binary',split_date=66,impute=True):

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
            fmiss = "f_43,f_51,f_58,f_59,f_64,f_65,f_66,f_67,f_68,f_69,f_70".split(',')
            for f in tqdm(fmiss,desc="NUM IMPUTE"):
                self.data[f].fillna(self.data[f].mean(),inplace=True)

    def get_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_split(self):
        if self.val_type == 'random':
            self.random_split()
        elif self.val_type == 'time':
            self.time_split()
        elif self.val_type == 'No':
            self.final_split()
        else:
            raise Exception('Invalid validation type')
    
    def get_label(self,data):
        '''
        data: Numpy array
        '''
        if self.class_type == 'binary':
            return data
        elif self.class_type == 'multi':
            labels = []
            for a, b in zip(data[:,0], data[:,1]):
                if a==0 and b==0:# None
                    labels.append(0)
                elif a==1 and b==0:# Clicked
                    labels.append(1)
                elif a==0 and b==1:# Installed
                    labels.append(2)
                elif a==1 and b==1:# Clicked and Installed
                    labels.append(3)
            return np.array(labels)

    def random_split(self):
        """
        Randomly split the data into train and test set
        """
        y = self.data[Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED].values
        X = self.data.drop(Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED, axis=1).values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def time_split(self):
        """
        Split the data into train and test set based on Date
        """     
        self.X_train = self.data[self.data[Train_Split.DATE] < self.split_date ].drop(Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED, axis=1).values
        self.X_test = self.data[self.data[Train_Split.DATE] >= self.split_date ].drop(Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED, axis=1).values
        self.y_train = self.get_label(self.data[self.data[Train_Split.DATE] < self.split_date ][Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED].values)
        self.y_test = self.get_label(self.data[self.data[Train_Split.DATE] >= self.split_date ][Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED].values)

    def final_split(self):
        self.X_train = self.data.drop(Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED, axis=1).values
        self.y_train = self.get_label(self.data[Train_Split.IS_CLICKED + Train_Split.IS_INSTALLED].values)
        self.X_test = None
        self.y_test = None




# class XGB_Model:
#     def __init__(self,X_train, X_test, y_train, y_test,params,use_features):
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.params = params
#         self.use_features = use_features

#     def train(self,X_train,y_train):
#         print("Training the model")
#         model = XGBClassifier(**self.params)
#         model.fit(X_train, y_train)
#         print("Training Done")
#         return model
    
#     def run_exp(self):
#         print("Starting the expirnment")
#         model = self.train(X_train=self.X_train,y_train=self.y_train)
#         y_pred = model.predict(self.X_test)
#         f1 = f1_score(self.y_test,y_pred)
#         print(f"F1 Score: {f1}")
#         return f1
    

    # def expirnment(self):
    #     print("Starting the expirnment")
    #     model = self.train(X_train=self.X_train,y_train=self.y_train)
    #     y_pred = model.predict(self.X_test)
    #     print("F1 Score: ",f1_score(self.y_test,y_pred,average='weighted'))
    #     full_model = self.train(X_train=np.concatenate((self.X_train,self.X_test),axis=0),
    #                             y_train=np.concatenate((self.y_train,self.y_test),axis=0))
    #     test = pd.read_csv('../Data/test/000000000000.csv',sep='\t')
    #     X = test[self.use_features].values
    #     y_pred = full_model.predict(X)
    #     return y_pred

def objective(trail):
    params = {
        'max_depth':trail.suggest_int('max_depth',3,10),
        'learning_rate':trail.suggest_loguniform('learning_rate',0.01,0.5),
        'n_estimators':trail.suggest_int('n_estimators',100,1000),
        'gamma':trail.suggest_loguniform('gamma',0.01,1),
        'colsample_bytree':trail.suggest_loguniform('colsample_bytree',0.01,1),
        'tree_method':'gpu_hist',
        'objective':'binary:logistic'
    }
    model = XGBClassifier(**params)
    print("Training the model")
    model.fit(X_train[use_features],y_train[:,1])
    print("Training Done")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test[:,1],y_pred)
    print(f"F1 Score: {f1}")
    return f1


if __name__=='__main__':
    train = Train_Split(val_type='time',class_type='multi',split_date=66,impute=True)
    X_train, X_test, y_train, y_test = train.get_split()

    use_features = Train_Split.CATEGORICAL + Train_Split.NUMERICAL + Train_Split.BINARY

