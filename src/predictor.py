import pandas as pd
import numpy as np
from src.model_utils import load_model, plot_feature_importance

def predict_pcos_risk(model, user_input):
    """
    Predict PCOS risk based on user health inputs.
    
    Args:
        model: Trained model
        user_input (dict): Dictionary containing user health data
    
    Returns:
        dict: Prediction results with probability and risk category
    """
    # Expected input keys: Age, BMI, Cycle_Length, Acne, Hair_Growth, Insulin, LH_FSH_Ratio, Weight_Gain, Irregular_Periods
    # Categorical values should be encoded as: Acne (0=No, 1=Yes), Hair_Growth (0=No, 1=Yes), etc.
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Normalize numerical features (assuming same scaler used in training)
    # For simplicity, we'll assume the model was trained with StandardScaler on these columns
    numerical_cols = ['Age', 'BMI', 'Cycle_Length', 'Insulin', 'LH_FSH_Ratio']
    
    # In a real scenario, you'd load the scaler used during training
    # For now, we'll use a simple normalization (this should match training preprocessing)
    for col in numerical_cols:
        if col in input_df.columns:
            # Using approximate normalization based on typical ranges
            # In production, use the actual scaler from training
            if col == 'Age':
                input_df[col] = (input_df[col] - 25) / 10  # Approximate normalization
            elif col == 'BMI':
                input_df[col] = (input_df[col] - 25) / 5
            elif col == 'Cycle_Length':
                input_df[col] = (input_df[col] - 28) / 7
            elif col == 'Insulin':
                input_df[col] = (input_df[col] - 10) / 5
            elif col == 'LH_FSH_Ratio':
                input_df[col] = (input_df[col] - 1.5) / 0.5
    
    # Get prediction probability
    prob = model.predict_proba(input_df)[0][1]  # Probability of positive class (PCOS risk)
    
    # Determine risk category
    if prob < 0.3:
        category = "Low"
    elif prob < 0.7:
        category = "Medium"
    else:
        category = "High"
    
    # Generate personalized health advice
    advice = generate_health_advice(user_input, category, prob)
    
    return {
        'probability': round(prob * 100, 2),
        'category': category,
        'advice': advice
    }

def generate_health_advice(user_input, risk_category, risk_probability):
    """
    Generate personalized health advice based on risk assessment and individual factors.
    
    Args:
        user_input (dict): User's health data
        risk_category (str): Low, Medium, or High risk
        risk_probability (float): Risk probability (0-1)
    
    Returns:
        dict: Personalized health recommendations
    """
    advice = {
        'general': [],
        'lifestyle': [],
        'monitoring': [],
        'when_to_see_doctor': [],
        'positive_message': ""
    }
    
    # Base advice by risk category
    if risk_category == "Low":
        advice['general'].append("Your risk appears low, but maintaining healthy habits is always beneficial.")
        advice['positive_message'] = "Great job maintaining your health! Continue with these wellness practices."
        
    elif risk_category == "Medium":
        advice['general'].append("You have a moderate risk. Focus on lifestyle changes and regular monitoring.")
        advice['monitoring'].append("Consider tracking your symptoms and menstrual cycles for 3-6 months.")
        advice['positive_message'] = "You're taking a positive step by assessing your health. Small changes can make a big difference!"
        
    else:  # High risk
        advice['general'].append("Your results suggest higher risk. Please consult a healthcare professional for proper evaluation.")
        advice['when_to_see_doctor'].append("Schedule an appointment with a gynecologist or endocrinologist within the next 2-4 weeks.")
        advice['positive_message'] = "Knowledge is power. Early awareness can lead to better health outcomes."
    
    # Personalized advice based on individual factors
    
    # BMI advice
    if user_input.get('BMI', 25) > 25:
        advice['lifestyle'].append("Consider gradual weight management through balanced diet and regular exercise.")
        advice['lifestyle'].append("Aim for 150 minutes of moderate aerobic activity per week.")
    
    # Insulin advice
    if user_input.get('Insulin', 10) > 12:
        advice['lifestyle'].append("Focus on low-glycemic foods and regular meal timing to support insulin sensitivity.")
        advice['monitoring'].append("Consider discussing insulin resistance testing with your doctor.")
    
    # Cycle length advice
    if user_input.get('Cycle_Length', 28) > 35 or user_input.get('Cycle_Length', 28) < 21:
        advice['monitoring'].append("Track your menstrual cycles to identify patterns and irregularities.")
    
    # Symptom-based advice
    if user_input.get('Acne', 0) == 1:
        advice['lifestyle'].append("Maintain good skincare routine and consider hormonal evaluation for acne management.")
    
    if user_input.get('Hair_Growth', 0) == 1:
        advice['lifestyle'].append("Consider professional evaluation for excess hair growth, which may be hormone-related.")
    
    if user_input.get('Weight_Gain', 0) == 1:
        advice['lifestyle'].append("Focus on sustainable weight management strategies rather than restrictive dieting.")
    
    if user_input.get('Irregular_Periods', 0) == 1:
        advice['monitoring'].append("Keep a menstrual calendar to track cycle patterns and discuss with your healthcare provider.")
    
    # LH/FSH ratio advice
    if user_input.get('LH_FSH_Ratio', 1.5) > 2.0:
        advice['monitoring'].append("Discuss hormone level testing with your doctor to understand your hormonal balance.")
    
    # Age-specific advice
    age = user_input.get('Age', 25)
    if age < 25:
        advice['general'].append("At your age, establishing healthy habits now can prevent future health concerns.")
    elif age > 35:
        advice['general'].append("Regular health screenings become increasingly important as you age.")
    
    # General wellness advice (always include)
    advice['lifestyle'].extend([
        "Aim for 7-9 hours of quality sleep per night.",
        "Stay hydrated with at least 8 glasses of water daily.",
        "Include stress management techniques like meditation or yoga.",
        "Eat a balanced diet rich in vegetables, fruits, whole grains, and lean proteins."
    ])
    
    # Remove duplicates
    for key in advice:
        if isinstance(advice[key], list):
            advice[key] = list(set(advice[key]))
    
    return advice

def get_feature_importance_plot(model, feature_names):
    """
    Generate feature importance plot.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_title("Feature Importances for PCOS Risk Prediction")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel("Importance")
    plt.tight_layout()
    
    return fig