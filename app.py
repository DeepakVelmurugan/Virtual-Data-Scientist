from flask import Flask, url_for, render_template, request, flash, redirect
from flask_sqlalchemy import SQLAlchemy
import csv
import pandas as pd

app = Flask(__name__)
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

@app.route('/upload',methods=['POST','GET'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['inputfile']
            df = pd.read_csv(request.files.get('inputfile'))
            #print(df)
            # newFile = FileContents(name=file.filename,data=file.read())
            # db.session.add(newFile)
            # db.session.commit()
            # csv_dicts = [{k: v for k, v in row.items()} for row in csv.DictReader(fstring.splitlines(), skipinitialspace=True)]
            # print(csv_dicts)
            return render_template('upload.html',  tables=[df.to_html(index=False, classes='data' , header='true')])
        except:
            return "Error saving your file (Check where the corresponding file is not empty):("

if __name__ == '__main__':
    app.run(debug=True)