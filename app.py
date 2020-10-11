from flask import Flask, url_for, render_template, request, flash, redirect,session
from flask_sqlalchemy import SQLAlchemy
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

@app.route('/train_test',methods=['POST','GET'])
def train_test():
    df  = pd.read_csv("Uploaded.csv")
    if request.method == "POST":
        df = df.drop(df.columns[[0]],axis=1)   #dropping first column which is of no use
        y = df.iloc[:,-1]                      #getting target variable
        percent = request.form.get("myRange")  #percentage split
        df.drop([y.name],axis=1,inplace=True)  #dropping target variable
        train_size = int(percent)/100   
        X_train,X_val,y_train,y_val = train_test_split(df,y,train_size=train_size,random_state=0) #splitting 
        print(y_train.head())
        return render_template('train_test.html')      
    return render_template('train_test.html')

@app.route('/data_preprocessing',methods=['POST','GET'])
def data_preprocessing():
    df = pd.read_csv("Uploaded.csv")
    if request.method == 'POST': 
        y_column_name_preprocessing = request.form.get('y_column_name_preprocessing')  # y_column_name
        select = request.form.get('drop_down')   #Type of missing
        df = df.drop(df.columns[[0]],axis=1)     #dropping first column which is of no use
        y = df[y_column_name_preprocessing]
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
        df_global.to_csv("Uploaded.csv")
        return render_template('data_preprocessing.html',tables=[df.head().to_html(index=False,classes='data'),
        y.head().to_html(index=False,classes='data')],titles=['na','Input Parameters','Target parameter'])
    return render_template('data_preprocessing.html')

@app.route('/upload',methods=['POST','GET'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['inputfile']
            df = pd.read_csv(request.files.get('inputfile'))
            df_global =  df.copy()
            df_global.to_csv("Uploaded.csv")
            dfcopy = df.head()
            #print(df)
            # newFile = FileContents(name=file.filename,data=file.read())
            # db.session.add(newFile)
            # db.session.commit()
            return render_template('upload.html',  tables=[dfcopy.to_html(index=False, classes='data' , header='true')])
        except:
            return "Error saving your file (Check where the corresponding file is not empty):("

if __name__ == '__main__':
    app.run(debug=True)
