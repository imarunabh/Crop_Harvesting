import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.preprocessing import LabelEncoder

# Load dataset (replace 'crop_data.csv' with your actual dataset)
df = pd.read_csv('Crop_recommendation.csv')

# Assume dataset has columns: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model (Support Vector Machine)
svm = SVC(kernel='linear', random_state=42)  # Using a linear kernel, but you can experiment with others
svm.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(svm, 'crop_svm_model.pkl')  # Saving the SVM model
joblib.dump(label_encoder, 'label_encoder.pkl')  # Saving the label encoder

print("SVM model and label encoder saved successfully!")
