import pandas as pd
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
data= pd.read_csv('Bengaluru_House_Data.csv') 
pipe=pickle.load(open('model.pkl','rb'))
# Example route
@app.route('/')
def index():
    location = sorted(data['location'].dropna().astype(str).unique())
    return render_template('index.html',locations=location)
@app.route('/predict', methods=['POST'])

def predict():
    try:
        location = request.form.get('location').strip()  # 👈 strip whitespace
        bhk = int(request.form.get('bhk'))
        bath = int(request.form.get('bath'))
        total_sqft = float(request.form.get('total_sqft'))

        input_df = pd.DataFrame([{
            'location': location,
            'total_sqft': total_sqft,
            'bath': bath,
            'bhk': bhk
        }])

        prediction = pipe.predict(input_df)[0]
        if prediction < 0:
            return str(0),500

        return str(round(prediction, 2))
    except Exception as e:
        print(f"Error during prediction: {e}")
        return str(e), 500



if __name__ == '__main__':
    app.run(debug=True,port=3000)
