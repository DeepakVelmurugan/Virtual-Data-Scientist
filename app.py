from flask import Flask, url_for, render_template, request, flash, redirect,session
from flask_sqlalchemy import SQLAlchemy
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

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
        dictionary = request.get_json()
        if dictionary is not None:
            df = df.drop(df.columns[[0]],axis=1) #dropping first column which is of no use
            y = df[session['val']]
            df.drop([session['val']],axis=1,inplace=True)  #dropping target variable
            train_size = int(dictionary['value'])/100   
            X_train,y_train,X_val,y_val = train_test_split(df,y,train_size=train_size,random_state=0)
            return render_template('train_test.html')      
        try:
            session['val'] = request.form['y_column_name'] #Session variable which is reusable
            y_column_name = session['val']
            df = df.drop(df.columns[[0]],axis=1) #dropping first column which is of no use
            y = df[y_column_name]
            df.drop([y_column_name],axis=1,inplace=True)  #dropping target variable
            return render_template('train_test.html',tables=[df.head().to_html(index=False,classes='data'),
                y.to_frame().head().to_html(index=False,classes='data')],titles=['na','Input Parameters','Target parameter'])
        except:
            return "Try checking for typos in y column name"
    return render_template('train_test.html')

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
