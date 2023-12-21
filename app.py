from flask import Flask, url_for, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # [total_hamil, nilai_glukosa, tekanan_darah, nilai_insulin, nilai_bmi, umur]
    
    umur = request.form.get('umur')
    total_hamil = request.form.get('total_hamil')
    nilai_glukosa = request.form.get('nilai_glukosa')
    tekanan_darah = request.form.get('tekanan_darah')
    nilai_insulin = request.form.get('nilai_insulin')
    nilai_bmi = request.form.get('nilai_bmi')

    features = [
        float(request.form.get('total_hamil')),
        float(request.form.get('nilai_glukosa')),
        float(request.form.get('tekanan_darah')),
        float(request.form.get('nilai_insulin')),
        float(request.form.get('nilai_bmi')),
        float(request.form.get('umur'))
    ]

    # Reshape the features to match the model's expectations
    features = np.array(features).reshape(1, -1)

    # Make predictions using the loaded model
    result = model.predict(features)

    # Convert the prediction (0 or 1) into a string
    hasil = 'Risiko Diabetes Tinggi' if result[0] == 1 else 'Risiko Diabetes Rendah'

    return render_template('index.html', hasil=hasil)

    return render_template('index.html', hasil='Bagus')

if __name__ == '__main__':
    app.run(debug=True)