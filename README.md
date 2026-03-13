# PCOS Risk Prediction Project

This is a machine learning project that predicts the risk of Polycystic Ovary Syndrome (PCOS) using health data and provides personalized health recommendations.

## Features

- **Data Preprocessing**: Handles missing values, normalizes numerical features, and encodes categorical variables
- **Machine Learning Model**: Random Forest classifier trained on synthetic PCOS dataset
- **Web Interface**: Streamlit app for easy prediction with personalized health advice
- **Health Recommendations**: Provides tailored lifestyle tips, monitoring guidance, and medical consultation advice based on individual risk factors
- **Feature Importance**: Visualization of which factors contribute most to PCOS risk prediction

## Dataset

The project uses a synthetic dataset containing the following features:
- Age
- BMI (Body Mass Index)
- Menstrual cycle length
- Acne presence
- Excessive hair growth
- Insulin level
- LH/FSH ratio
- Weight gain
- Irregular periods

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Create Dataset** (if needed):
   ```bash
   python create_dataset.py
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```

3. **Run the Web App**:
   ```bash
   streamlit run app.py
   ```

## Health Advice Features

The app now provides personalized health recommendations based on:

- **Risk Category**: Low, Medium, or High risk assessment
- **Individual Factors**: BMI, insulin levels, symptoms, and hormonal markers
- **Lifestyle Tips**: Exercise, diet, and wellness recommendations
- **Monitoring Guidance**: When and what to track
- **Medical Consultation**: When to seek professional help

### Advice Categories:
- **General Recommendations**: Overall health guidance
- **Lifestyle & Wellness Tips**: Daily habits and routines
- **Monitoring & Tracking**: Symptom and cycle tracking
- **When to See a Doctor**: Medical consultation guidance
- **Positive Messages**: Encouraging feedback based on risk level

## Project Structure

```
├── data/
│   └── pcos_dataset.csv          # Synthetic dataset
├── src/
│   ├── model_utils.py            # Model training and evaluation utilities
│   └── predictor.py              # Prediction functions and health advice
├── models/
│   ├── pcos_model.pkl            # Trained model
│   └── feature_importance.png    # Feature importance plot
├── app.py                        # Streamlit web application
├── train_model.py                # Model training script
├── create_dataset.py             # Dataset creation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Model Evaluation

The Random Forest model is evaluated using:
- Accuracy
- Precision
- Recall
- Confusion Matrix

## Disclaimer

This tool is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical concerns.

## License

This project is open source and available under the MIT License.