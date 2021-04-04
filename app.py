from flask import Flask, url_for, render_template, request, flash, redirect,session
from flask_sqlalchemy import SQLAlchemy
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.impute import SimpleImputer
import os.path
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model,tree
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,f1_score,accuracy_score
from sklearn.neural_network import MLPClassifier,MLPRegressor
import os

app = Flask(__name__)
app.secret_key = "any random string"
app.config['SQLALCHEMY_DATABASE_URI']  = 'sqlite:///test_file_upload.db'
db = SQLAlchemy(app)

class FileContents(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

#Delete all csv files
def remove_csv():
    for folder, subfolders, files in os.walk('csv/'):            
        for file in files:            
            # checking if file is  
            # of .txt type  
            if file.endswith('.csv'):  
                path = os.path.join(folder, file)                    
                # printing the path of the file  
                # to be deleted  
                print('deleted : ', path )                
                # deleting the csv file  
                os.remove(path)
    return "done"

#ML Model Page
#lambda function for helping with none
helper_none = lambda x : None if x == 'None' else int(x)
#RandomForestRegressor
def randomForestRegressor(dt):
    n_estimators = dt.get('n_estimators',100)
    max_depth = helper_none(dt.get('max_depth'))
    random_state = helper_none(dt.get('random_state'))
    model = RandomForestRegressor(n_estimators=int(n_estimators[0]),max_depth= max_depth,random_state=random_state)
    return model

#lambda function for helping with bool
helper_bool = lambda x : True if x == "True" else False

#LinearRegression
def linearRegression(dt):
    intercept = helper_bool(dt.get('intercept'))
    normalize = helper_bool(dt.get('normalize'))
    model = linear_model.LinearRegression(fit_intercept=intercept,normalize=normalize)
    return model

#LogisticRegression
def logisticRegression(dt):
     penalty = dt.get("penalty")
     max_iter = helper_none(dt.get("max_iter"))
     random_state = helper_none(dt.get("random_state"))
     model = linear_model.LogisticRegression(penalty=penalty,max_iter=max_iter,random_state=random_state)
     return model

#XGB
def xtremeGradient(dt):
    n_estimators = helper_none(dt.get("n_estimators"))
    max_depth = helper_none(dt.get("max_depth"))
    random_state = helper_none(dt.get("random_state"))
    learning_rate = float(dt.get("learning_rate"))
    model = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state,learning_rate=learning_rate)
    return model

#DecisionTreeClassifier
def decisionTreeClassifier(dt):
    max_features = helper_none(dt.get("max_features"))
    max_depth = helper_none(dt.get("max_depth"))
    random_state = helper_none(dt.get("random_state"))
    model = tree.DecisionTreeClassifier(max_features=max_features,max_depth=max_depth,random_state=random_state)
    return model

#NeuralNets
def neuralNet(dt):
    hidden_layers = tuple(map(int,dt.get("hidden_layers").split(",")))
    activation =  dt.get("activation")
    solver = dt.get("solver")
    alpha = float(dt.get("learning_rate"))
    nesterovs_momentum = dt.get('nesterovs_momentum','False')
    nesterovs_momentum = True if nesterovs_momentum == "True" else False
    type_of_model = dt.get("type_of_model")
    if type_of_model == "Classification":
        model = MLPClassifier(hidden_layer_sizes=hidden_layers,activation=activation,solver=solver,alpha=alpha,nesterovs_momentum=nesterovs_momentum)
    else:
        model = MLPRegressor(hidden_layer_sizes=hidden_layers,activation=activation,solver=solver,alpha=alpha,nesterovs_momentum=nesterovs_momentum)
    return model

def model_helper(model,test_dt):
    if test_dt["type"] == "randomforest":
        model = randomForestRegressor(test_dt)
    elif test_dt["type"] == "logisticregression":
        model = logisticRegression(test_dt)
    elif test_dt["type"] == "LinearRegression":
        model = linearRegression(test_dt)
    elif test_dt["type"] == "XGBoost":
        model = xtremeGradient(test_dt)
    elif test_dt["type"] == "DecisionTreeClassifier":
        model = decisionTreeClassifier(test_dt)
    elif test_dt["type"] == "NeuralNets":
        model = neuralNet(test_dt)
    return model

def loss_helper(test_dt,y_val,pred):
    loss = None
    try:
        if test_dt["evaluation_metric"] == "MAE":
            loss = mean_absolute_error(y_val,pred)
        elif test_dt["evaluation_metric"] == "MSE":
            loss = mean_squared_error(y_val,pred,squared=False)
        elif test_dt["evaluation_metric"] == "F1":
            loss = f1_score(y_val,pred)
        elif test_dt["evaluation_metric"] == "Accuracy":
            loss = accuracy_score(y_val,pred)
    except:
        return 'Error ,please give F1 or Accuracy metric for binary classification!'
    return loss

@app.route('/MLmodels',methods=['POST','GET'])
def MLmodels():
    try:
        if request.method == "POST":
            test_dt = request.form.to_dict(flat=False)
            test_dt = {k:v[0] for k,v in test_dt.items()}         #Preprocessing dictionary from form
            model = None
            if(os.path.isfile("csv/train.csv") and os.path.isfile("csv/val.csv")): #if oridinary train-test split
                X_train = pd.read_csv("csv/train.csv")            
                X_train = X_train.drop(X_train.columns[[0]],axis=1)             #dropping first column which is of no use
                y_train = X_train.iloc[:,-1]                          #y assignment
                X_train.drop([y_train.name],axis=1,inplace=True)      #dropping y 
                X_val = pd.read_csv("csv/val.csv")
                X_val = X_val.drop(X_val.columns[[0]],axis=1)         #dropping first column which is of no use
                y_val = X_val.iloc[:,-1]
                X_val.drop([y_val.name],axis=1,inplace=True)
                model = model_helper(model,test_dt)
                model.fit(X_train,y_train)
                pred = model.predict(X_val)  
                loss = loss_helper(test_dt,y_val,pred)
                return render_template("MLmodels.html",scores=[loss])
            elif os.path.isfile("csv/splitted.csv"):
                df = pd.read_csv("csv/splitted.csv")
                X = df.drop(df.columns[[0]],axis=1)
                y = df.iloc[:,-1]
                X.drop([y.name],axis=1,inplace=True)
                no_of_folds = X["kfold"].max()+1          
                model = model_helper(model,test_dt)
                loss_list = []
                for fold in range(no_of_folds):
                    X_train = X[df.kfold != fold]
                    X_val = X[df.kfold == fold]
                    y_train = y[df.kfold != fold]
                    y_val = y[df.kfold == fold]
                    model.fit(X_train,y_train)
                    pred = model.predict(X_val)
                    val = loss_helper(test_dt,y_val,pred)
                    if isinstance(val,str):
                        return render_template('MLmodels.html',scores=[val])
                    loss_list.append(val)
                scores = sum(loss_list)/no_of_folds
                return render_template('MLmodels.html',scores=[scores])
            return render_template('MLmodels.html',scores=[])
        return render_template('MLmodels.html',scores=[])
    except:
        return '<h1 style="background: red;">Error occured while predicting!</h1>'

#Train-test page
@app.route('/train_test',methods=['POST','GET'])
def train_test():
    df = pd.read_csv("csv/Uploaded.csv")
    try:
        if request.method == "POST":
            df = df.drop(df.columns[[0]],axis=1)   #dropping first column which is of no use
            y = df.iloc[:,-1]                      #getting target variable 
            df.drop([y.name],axis=1,inplace=True)  #dropping target variable
            percent = request.form.get("myRange")  #percentage split general train-test split
            kfold = request.form.get("drop_down")  #No of splits for kfold
            kfold_st = request.form.get("drop_down_st") #No of splits for Stratified kfold
            kfold_rg = request.form.get("drop_down_rg") #No of splits for Stratified kfold regression
            if percent is not None:
                train_size = int(percent)/100   
                X_train,X_val,y_train,y_val = train_test_split(df,y,train_size=train_size,random_state=0) #splitting
                df_train = pd.merge(X_train,y_train,left_index=True,right_index=True)
                df_train.to_csv("csv/train.csv")
                df_val = pd.merge(X_val,y_val,left_index=True,right_index=True)
                df_val.to_csv("csv/val.csv")
            elif kfold is not None:
                df["kfold"] = -1
                sh = request.form.get("shuffle_dataset") #getting shuffle variable
                random_state = request.form.get("random_state") #getting random state
                shuffle_flag = False
                if sh == "shuffle": 
                    shuffle_flag = True
                kf = KFold(n_splits = int(kfold),shuffle=shuffle_flag,random_state=helper_none(random_state))  #Kfold split
                for fold,(trn_,val_) in enumerate(kf.split(X=df)):
                    df.loc[val_,'kfold'] = fold
                y = y.to_frame()
                df_global = pd.merge(df,y,left_index=True,right_index=True)
                df_global.to_csv("csv/splitted.csv")
            elif kfold_st is not None:
                df["kfold"] = -1
                sh = request.form.get("shuffle_dataset_st") #getting shuffle variable
                random_state = request.form.get("random_state_st") #getting random state
                shuffle_flag = False
                if sh == "shuffle":
                    shuffle_flag = True             
                kf = StratifiedKFold(n_splits = int(kfold_st),shuffle=shuffle_flag,random_state=helper_none(random_state)) #KFold split without random_state
                for fold,(trn_,val_) in enumerate(kf.split(X=df,y=y)):
                    df.loc[val_,'kfold'] = fold
                y = y.to_frame()
                df_global = pd.merge(df,y,left_index=True,right_index=True)
                df_global.to_csv("csv/splitted.csv")

            elif kfold_rg is not None:
                df["kfold"] = -1
                num_bins = int(np.floor(1+np.log2(len(df)))) #No of categories
                df.loc[:,"bins"] = pd.cut(y,bins=num_bins,labels=False) #Creating category column
                sh = request.form.get("shuffle_dataset_rg") #getting shuffle variable
                random_state = request.form.get("random_state_rg") #getting random state
                shuffle_flag = False
                if sh == "shuffle":
                    shuffle_flag = True
                kf = StratifiedKFold(n_splits = int(kfold_rg),shuffle=shuffle_flag,random_state=helper_none(random_state)) #KFold split without random_state
                for fold,(trn_,val_) in enumerate(kf.split(X=df,y=df.bins.values)):
                    df.loc[val_,'kfold'] = fold
                y = y.to_frame()
                df_global = pd.merge(df,y,left_index=True,right_index=True)
                df_global.to_csv("csv/splitted.csv")

            return render_template('train_test.html',  tables=[df.head().to_html(index=False, classes='data' , header='true')])
    except:
        return '<h1 style="background: red;">Error occured while splitting!</h1>'

    return render_template('train_test.html')

#Data-preprocess page
@app.route('/data_preprocessing',methods=['POST','GET'])
def data_preprocessing():
    df = pd.read_csv("csv/Uploaded.csv")
    try:
        if request.method == 'POST': 
            y_column_name_preprocessing = request.form.get('y_column_name_preprocessing')  # y_column_name
            select = request.form.get('drop_down')   #Type of missing
            df = df.drop(df.columns[[0]],axis=1)     #dropping first column which is of no use
            try:
                y = df[y_column_name_preprocessing]
            except:
                return '<h1 style="background: red;">Column name not present!</h1>'
            df.dropna(axis=0,subset=[y_column_name_preprocessing],inplace=True) #dropping rows having unknown target variable
            df.drop([y_column_name_preprocessing],axis=1,inplace=True)  #dropping target variable
            X = df.select_dtypes(exclude=['object'])  #Removing categorical data as of now
            if select == "drop":                      #Removing missing columns
                cols_missing_vals = [col for col in X.columns if X[col].isnull().any()]
                X_pre = X.drop(cols_missing_vals,axis=1)
            elif select == "impute":                   #Imputing missing columns
                imputer = SimpleImputer()
                X_pre = pd.DataFrame(imputer.fit_transform(X))
                X_pre.columns = X.columns
            elif select == "extension":                #Extension to imputing
                imputer = SimpleImputer()
                X_tmp = X.copy()
                cols_missing_vals = [col for col in X.columns if X[col].isnull().any()]
                for col in cols_missing_vals:
                    X_tmp[col + '_was_missing'] = X_tmp[col].isnull()
                X_pre = pd.DataFrame(imputer.fit_transform(X_tmp))
                X_pre.columns = X_tmp.columns
            df= X_pre.copy() 
            #Combining samples and target variable for splitting
            y = y.to_frame()
            df_global = pd.merge(df,y,left_index=True,right_index=True)
            df_global.to_csv("csv/Uploaded.csv")
            return render_template('data_preprocessing.html',tables=[df.head().to_html(index=False,classes='data'),
            y.head().to_html(index=False,classes='data')],titles=['na','Input Parameters','Target parameter'])
        return render_template('data_preprocessing.html')
    except:
        return '<h1 style="background: red;">Error occured while preprocessing data</h1>'

#Upload page
@app.route('/upload',methods=['POST','GET'])
def upload_file():
    if request.method == 'POST':
        try:       
            idx_col = request.form.get("idx")
            if idx_col != "None":
                file = request.files['inputfile']
                df = pd.read_csv(request.files.get('inputfile'),index_col=idx_col)
                df_global =  df.copy()
                isDeleted = remove_csv()
                df_global.to_csv("csv/Uploaded.csv")
                dfcopy = df.head()
                if len(dfcopy.columns) < 5:
                    dfcopy = dfcopy.iloc[:,0:len(dfcopy.columns)]
                dfcopy = dfcopy.iloc[:,0:5]
                #print(df)
                # newFile = FileContents(name=file.filename,data=file.read())
                # db.session.add(newFile)
                # db.session.commit()
                return render_template('upload.html',  tables=[dfcopy.to_html(index=False, classes='data' , header='true')])
            else:
                return render_template('upload.html')
        except:
           return '<h1 style="background: red;">Error ,please check type of file and index column name!</h1>'
    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
