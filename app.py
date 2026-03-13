import streamlit as st
import pandas as pd
import numpy as np
from src.model_utils import load_model
from src.predictor import predict_pcos_risk, get_feature_importance_plot

def main():
    st.title("PCOS Risk Prediction System")
    st.write("Enter your health information to assess your risk of Polycystic Ovary Syndrome (PCOS).")
    
    # Load the trained model
    try:
        model = load_model('models/pcos_model.pkl')
        feature_names = ['Age', 'BMI', 'Cycle_Length', 'Acne', 'Hair_Growth', 'Insulin', 'LH_FSH_Ratio', 'Weight_Gain', 'Irregular_Periods']
    except FileNotFoundError:
        st.error("Model not found. Please train the model first by running the training script.")
        return
    
    # Create input form
    st.header("Health Information Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=25)
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=22.0, step=0.1)
        cycle_length = st.number_input("Menstrual Cycle Length (days)", min_value=20, max_value=45, value=28)
        insulin = st.number_input("Insulin Level", min_value=0.0, max_value=50.0, value=8.0, step=0.1)
        lh_fsh_ratio = st.number_input("LH/FSH Ratio", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    
    with col2:
        acne = st.selectbox("Acne", ["No", "Yes"])
        hair_growth = st.selectbox("Excessive Hair Growth", ["No", "Yes"])
        weight_gain = st.selectbox("Recent Weight Gain", ["No", "Yes"])
        irregular_periods = st.selectbox("Irregular Periods", ["No", "Yes"])
    
    # Convert categorical inputs to numerical
    acne_num = 1 if acne == "Yes" else 0
    hair_growth_num = 1 if hair_growth == "Yes" else 0
    weight_gain_num = 1 if weight_gain == "Yes" else 0
    irregular_periods_num = 1 if irregular_periods == "Yes" else 0
    
    # Prepare user input
    user_input = {
        'Age': age,
        'BMI': bmi,
        'Cycle_Length': cycle_length,
        'Acne': acne_num,
        'Hair_Growth': hair_growth_num,
        'Insulin': insulin,
        'LH_FSH_Ratio': lh_fsh_ratio,
        'Weight_Gain': weight_gain_num,
        'Irregular_Periods': irregular_periods_num
    }
    
    # Prediction button
    if st.button("Predict PCOS Risk"):
        result = predict_pcos_risk(model, user_input)
        
        st.header("🩺 Prediction Results")
        
        # Display probability
        st.subheader(f"PCOS Risk Probability: {result['probability']}%")
        
        # Display risk category with color coding
        if result['category'] == "Low":
            st.success(f"Risk Category: {result['category']}")
            st.write("Your risk of PCOS appears to be low. However, consult with a healthcare professional for personalized advice.")
        elif result['category'] == "Medium":
            st.warning(f"Risk Category: {result['category']}")
            st.write("You may have a moderate risk of PCOS. Consider consulting a healthcare professional for further evaluation.")
        else:
            st.error(f"Risk Category: {result['category']}")
            st.write("You appear to have a high risk of PCOS. Please consult with a healthcare professional for proper diagnosis and treatment.")
        
        # Display personalized health advice
        st.header("💡 Personalized Health Recommendations")
        
        advice = result['advice']
        
        # Positive message
        if advice['positive_message']:
            st.info(f"✨ {advice['positive_message']}")
        
        # General advice
        if advice['general']:
            st.subheader("📋 General Recommendations")
            for item in advice['general']:
                st.write(f"• {item}")
        
        # Lifestyle advice
        if advice['lifestyle']:
            st.subheader("🏃‍♀️ Lifestyle & Wellness Tips")
            for item in advice['lifestyle']:
                st.write(f"• {item}")
        
        # Monitoring advice
        if advice['monitoring']:
            st.subheader("📊 Monitoring & Tracking")
            for item in advice['monitoring']:
                st.write(f"• {item}")
        
        # When to see doctor
        if advice['when_to_see_doctor']:
            st.subheader("🏥 When to Consult a Healthcare Professional")
            for item in advice['when_to_see_doctor']:
                st.write(f"• {item}")
            st.warning("**Important:** These recommendations are for informational purposes only and should not replace professional medical advice.")
    
    # Feature importance visualization
    st.header("📈 Feature Importance")
    st.write("This chart shows which health factors are most important in predicting PCOS risk.")
    
    fig = get_feature_importance_plot(model, feature_names)
    st.pyplot(fig)
    
    # Disclaimer
    st.header("Important Disclaimer")
    st.write("""
    This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. 
    Never disregard professional medical advice or delay in seeking it because of something you have read here.
    """)

if __name__ == "__main__":
    main()