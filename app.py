from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# โหลดโมเดล ML
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # ดึงข้อมูลจากฟอร์ม
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    thalach = int(request.form['thalach'])
    
    # สร้าง DataFrame เพื่อใช้ในการทำนาย
    data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, thalach]], 
                        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach'])
    
    # ทำการทำนาย
    prediction = model.predict(data)[0]
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
