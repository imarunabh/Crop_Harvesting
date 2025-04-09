import serial
import time
import joblib
import requests
import geocoder
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained ML model and label encoder
model = joblib.load('crop_svm_model.pkl')  # Replace with actual path
label_encoder = joblib.load('label_encoder.pkl')  # Replace with actual path

def get_location():
    g = geocoder.ip("me")
    if g.ok and g.latlng:
        return g.latlng
    return None, None

def get_annual_humidity(latitude, longitude):
    try:
        api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=relative_humidity_2m_max&past_days=7&timezone=auto"
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        humidity_values = data.get("daily", {}).get("relative_humidity_2m_max", [])
        if not humidity_values:
            return "Data Unavailable"
        return round(sum(humidity_values) / len(humidity_values), 2)
    except requests.exceptions.RequestException:
        return "Data Unavailable"

def read_ph_from_arduino(port='COM7', baud_rate=9600):
    try:
        ser = serial.Serial(port, baud_rate, timeout=2)
        time.sleep(2)
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if "pH Value" in line:
                    try:
                        ph_part = line.split('|')[-1]
                        ph_value = float(ph_part.split(':')[-1].strip())   # Calibration offset
                        ser.close()
                        return round(ph_value, 2)
                    except (IndexError, ValueError):
                        continue
    except serial.SerialException:
        return None

@app.route('/fetch_ph', methods=['GET'])
def fetch_ph():
    ph_value = read_ph_from_arduino()
    if ph_value is not None:
        return jsonify({"success": True, "ph": ph_value})
    else:
        return jsonify({"success": False, "error": "Failed to read pH"}), 500

@app.route('/fetch_weather', methods=['GET'])
def fetch_weather():
    latitude, longitude = get_location()
    if latitude is None or longitude is None:
        return jsonify({"success": False, "error": "Could not determine location"}), 400

    humidity = get_annual_humidity(latitude, longitude)
    nasa_api_url = f"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=PRECTOTCORR,T2M&community=RE&longitude={longitude}&latitude={latitude}&format=JSON"
    try:
        response = requests.get(nasa_api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        rainfall_per_day = data.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {}).get("ANN")
        temperature = data.get("properties", {}).get("parameter", {}).get("T2M", {}).get("ANN")
        rainfall_per_year = round(rainfall_per_day * 365, 2) if rainfall_per_day is not None else "Data Unavailable"
        return jsonify({
            "success": True,
            "rainfall": rainfall_per_year,
            "temperature": temperature if temperature is not None else "Data Unavailable",
            "humidity": humidity
        })
    except requests.exceptions.RequestException:
        return jsonify({"success": False, "error": "NASA API error"}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        ph = float(request.form.get("ph"))
        rainfall = float(request.form.get("rainfall") or 0)
        temperature = float(request.form.get("temperature") or 0)
        humidity = float(request.form.get("humidity") or 0)

        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(input_data)
        prediction_result = label_encoder.inverse_transform(prediction)[0]

        return render_template("result.html", prediction=prediction_result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
