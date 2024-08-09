from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import threading
import base64
import re
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

models_dir = os.path.join(os.path.dirname(__file__), 'Models')
models = {
    'NAFLD': joblib.load(os.path.join(models_dir, 'best_model_nafld.pkl')),
    'ALBI': joblib.load(os.path.join(models_dir, 'best_model_albi.pkl')),
    'LFT': joblib.load(os.path.join(models_dir, 'best_model_lft.pkl'))
}

scaler = StandardScaler()

progress = 0
results = {}

def preprocess_image(image):
    image = image.convert('L')
    image = ImageEnhance.Contrast(image).enhance(2)
    image = ImageEnhance.Sharpness(image).enhance(2)
    image = image.filter(ImageFilter.MedianFilter())
    return image

def correct_ocr_errors(text):
    text = text.replace('aw', '1.1')
    text = text.replace('WW', '1.1')
    return text

def extract_lft_values(image):
    image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    text = correct_ocr_errors(text)
    print("Extracted Text:\n", text)

    lines = text.split('\n')

    values = {
        "Age of the patient": None,
        "Total Bilirubin": None,
        "ALB Albumin": None,
        "Gender": 'Male',
        "Sgpt Alamine Aminotransferase": None,
        "Sgot Aspartate Aminotransferase": None,
        "Direct Bilirubin": None,
        "Alkphos Alkaline Phosphotase": None,
        "Total Protiens": None,
        "A/G Ratio Albumin and Globulin Ratio": None
    }

    for line in lines:
        line = line.strip()
        if "Age" in line:
            age_match = re.search(r'(\d+)', line)
            values["Gender"] = 0
            if age_match:
                values["Age of the patient"] = int(age_match.group(1))
        elif "BILIRUBIN TOTAL" in line:
            bil_match = re.search(r'(\d+\.\d+|\d+)', line)
            if bil_match:
                values["Total Bilirubin"] = float(bil_match.group(1))
        elif "ALBUMIN" in line:
            alb_match = re.search(r'(\d+\.\d+|\d+)', line)
            if alb_match:
                values["ALB Albumin"] = float(alb_match.group(1))
        elif "‘SGPT" in line:
            sgpt_match = re.search(r'(\d+\.\d+|\d+)', line)
            if sgpt_match:
                values["Sgpt Alamine Aminotransferase"] = float(sgpt_match.group(1))
        elif "SGOT" in line and "SGPT" not in line:
            sgot_match = re.search(r'(\d+\.\d+|\d+)', line)
            if sgot_match:
                values["Sgot Aspartate Aminotransferase"] = float(sgot_match.group(1))
        elif "DIRECT BILIRUBIN" in line:
            dir_bil_match = re.search(r'(\d+\.\d+|\d+)', line)
            if dir_bil_match:
                values["Direct Bilirubin"] = float(dir_bil_match.group(1))
        elif "ALKPHOS ALKALINE PHOSPHOTASE" in line:
            alkphos_match = re.search(r'(\d+\.\d+|\d+)', line)
            if alkphos_match:
                values["Alkphos Alkaline Phosphotase"] = float(alkphos_match.group(1))
        elif "TOTAL PROTIENS" in line:
            tp_match = re.search(r'(\d+\.\d+|\d+)', line)
            if tp_match:
                values["Total Protiens"] = float(tp_match.group(1))
        elif "A/G RATIO" in line:
            ag_ratio_match = re.search(r'(\d+\.\d+|\d+)', line)
            if ag_ratio_match:
                values["A/G Ratio Albumin and Globulin Ratio"] = float(ag_ratio_match.group(1))

    values = {k: v for k, v in values.items() if v is not None}

    print("Extracted Values:", values)
    return values

def preprocess_data(values, feature_columns):
    if any(col not in values for col in feature_columns):
        print("Missing values in the extracted data. Preprocessing aborted.")
        return None

    data = pd.DataFrame([values])
    print("Data before scaling:", data)
    data_scaled = scaler.fit_transform(data[feature_columns])
    print("Data after scaling:", data_scaled)
    return data_scaled

def process_file(function, file_data):
    global progress, results
    progress = 0

    try:
        image = Image.open(io.BytesIO(base64.b64decode(file_data.split(',')[1])))
        progress = 30

        feature_columns = {
            'NAFLD': ['Gender','Age of the patient', 'Total Bilirubin', 'ALB Albumin', 'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase'],
            'ALBI': ['Total Bilirubin', 'ALB Albumin'],
            'LFT': ['Total Bilirubin', 'Direct Bilirubin', 'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin', 'A/G Ratio Albumin and Globulin Ratio']
        }

        if function in feature_columns:
            values = extract_lft_values(image)
            data = preprocess_data(values, feature_columns[function])
            if data is None:
                results = {"error": "Not enough data for prediction"}
                progress = 100
                return

            model = models[function]
            prediction = model.predict(data)
            progress = 100

            results = {"prediction": prediction.tolist()}
        else:
            results = {"error": "Unsupported function"}
            progress = 100
    except Exception as e:
        results = {"error": str(e)}
        progress = 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    function = request.form.get('function')
    file = request.files.get('file')

    if not function or not file:
        return jsonify({"error": "Invalid input"}), 400

    if function not in models:
        return jsonify({"error": "Invalid function name"}), 400

    file_data = file.read()
    file_data_base64 = base64.b64encode(file_data).decode('utf-8')
    file_data_base64 = f"data:{file.content_type};base64,{file_data_base64}"

    threading.Thread(target=process_file, args=(function, file_data_base64)).start()
    return jsonify({"message": "Processing started"})

@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify({"progress": progress})

@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
