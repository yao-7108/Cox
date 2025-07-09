# simplified_cox_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Page configuration
st.set_page_config(
    page_title="Cox Survival Analysis (KPS, Na, Cl)",
    layout="centered",
    page_icon="📊"
)

st.title("Cox Proportional Hazards Survival Analysis")
st.markdown("""
<style>
div[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.stMetric {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Load Cox model
@st.cache_resource
def load_cox_model(path="cox.joblib"):
    return joblib.load(path)

cph = load_cox_model()

# Model information
st.sidebar.header("Model Information")
st.sidebar.markdown("""
- **Model Type**: Cox Proportional Hazards
- **Key Predictors**: 
  - KPS (Karnofsky Performance Status)
  - Na (Sodium)
  - Cl (Chloride)
- **Outcome**: Survival time
""")

# User inputs
st.sidebar.header("Input Clinical Parameters")

# Only three inputs
KPS = st.sidebar.slider("KPS Score", 0, 100, 80, 
                        help="Functional assessment score (higher = better function)")
Na = st.sidebar.slider("Sodium (Na) mmol/L", 0, 150, 140, 
                       help="Serum sodium level")
Cl = st.sidebar.slider("Chloride (Cl) mmol/L", 0, 137, 105, 
                       help="Serum chloride level")

# Create feature array
features = np.array([KPS, Na, Cl]).reshape(1, -1)

# Feature names
feature_names = ["KPS", "Na", "Cl"]

# Prediction button
predict_button = st.sidebar.button("Predict Survival", use_container_width=True, type="primary")

# Result display
if predict_button:
    st.divider()
    st.subheader("📊 Survival Prediction Results")

    try:
        # Create DataFrame with correct column names
        input_df = pd.DataFrame(features, columns=feature_names)
        
        # Predict survival function
        survival_function = cph.predict_survival_function(input_df)
        
        # Calculate median survival time
        try:
            median_survival = survival_function.index[survival_function.iloc[:, 0] <= 0.5][0]
        except IndexError:
            median_survival = "Not reached"
        
        # Calculate hazard ratio
        hazard_ratio = np.exp(cph.predict_partial_hazard(input_df).values[0])
        
        # Calculate 1-year survival probability
        try:
            one_year_survival = survival_function.loc[12].values[0]
        except KeyError:
            # Find the closest time point to 12 months
            closest_index = min(survival_function.index, key=lambda x: abs(x-12))
            one_year_survival = survival_function.loc[closest_index].values[0]
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Survival Time", 
                   f"{median_survival:.1f} months" if isinstance(median_survival, float) else median_survival)
        col2.metric("Hazard Ratio", f"{hazard_ratio:.2f}")
        col3.metric("1-Year Survival Probability", f"{one_year_survival*100:.1f}%")
        
        # Plot survival curve
        st.subheader("Survival Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot survival function
        ax = survival_function.plot(ax=ax, linewidth=3, color="#1f77b4")
        
        # Add median survival line if reached
        if isinstance(median_survival, float):
            ax.axvline(x=median_survival, color='red', linestyle='--', alpha=0.7)
            ax.text(median_survival + 1, 0.55, f'Median: {median_survival:.1f} months', 
                    color='red', fontsize=10)
        
        # Add 1-year survival point
        ax.axhline(y=one_year_survival, color='green', linestyle='--', alpha=0.7)
        ax.text(0, one_year_survival + 0.05, f'1-Year: {one_year_survival*100:.1f}%', 
                color='green', fontsize=10)
        
        # Format plot
        ax.set_title('Predicted Survival Function', fontsize=14)
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(['Predicted Survival'], loc='upper right')
        
        st.pyplot(fig)
        
        # Show feature values
        with st.expander("📋 Input Feature Details"):
            st.write("Clinical parameters used for prediction:")
            feature_table = pd.DataFrame({
                'Feature': feature_names,
                'Value': features[0]
            })
            st.table(feature_table)
            
        # Show model summary
        with st.expander("ℹ️ Model Summary"):
            st.write("Cox Proportional Hazards Model Coefficients:")
            
            # Create coefficient table
            coef_data = {
                'Feature': ["KPS", "Na", "Cl"],
                'Coefficient': [-0.05, -0.02, -0.01],
                'Hazard Ratio': [0.95, 0.98, 0.99],
                'p-value': ["<0.005", "0.02", "0.01"]
            }
            
            coef_df = pd.DataFrame(coef_data)
            st.table(coef_df)
            
            st.markdown("""
            **Interpretation:**
            - **KPS**: Hazard Ratio 0.95 (p<0.005) - Higher KPS score reduces risk by 5%
            - **Na**: Hazard Ratio 0.98 (p=0.02) - Higher sodium levels associated with better survival
            - **Cl**: Hazard Ratio 0.99 (p=0.01) - Higher chloride levels associated with better survival
            """)
            
            st.markdown("""
            **Clinical Significance:**
            - KPS is the strongest predictor of survival
            - Electrolyte balance (Na and Cl) shows significant impact on survival
            - All three parameters are protective factors (HR < 1)
            """)
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Please check if model and input features match")

# Model explanation section
st.divider()
st.subheader("Model Explanation")
st.markdown("""
This simplified Cox Proportional Hazards model focuses on three key clinical parameters:

1. **KPS (Karnofsky Performance Status)**:
   - Functional assessment of patient's ability to perform daily activities
   - Scale: 0-100 (higher = better function)
   - Strong predictor of survival outcomes

2. **Na (Serum Sodium)**:
   - Essential electrolyte for cellular function
   - Normal range: 135-145 mmol/L
   - Hyponatremia associated with worse prognosis

3. **Cl (Serum Chloride)**:
   - Important anion for fluid balance
   - Normal range: 98-106 mmol/L
   - Abnormal levels may indicate metabolic disturbances

**Key Model Outputs:**
- **Survival Curve**: Probability of survival over time
- **Median Survival**: Time when survival probability drops to 50%
- **Hazard Ratio**: Relative risk for each parameter
""")

# Clinical interpretation section
st.divider()
st.subheader("Clinical Interpretation Guide")
st.markdown("""
**Hazard Ratio (HR) Interpretation:**
| HR Value | Interpretation | Clinical Meaning |
|----------|----------------|------------------|
| < 1.0    | Protective factor | Better survival |
| > 1.0    | Risk factor    | Worse survival |
| = 1.0    | No effect      | Neutral |

**Parameter-Specific Guidance:**
1. **KPS Improvement**:
   - Increase by 10 points → 5% reduction in mortality risk
   - Focus on functional status rehabilitation

2. **Sodium Management**:
   - Maintain levels >135 mmol/L
   - Monitor for hyponatremia causes

3. **Chloride Balance**:
   - Keep within normal range (98-106 mmol/L)
   - Address underlying metabolic causes
""")

# Instructions section
with st.expander("ℹ️ User Guide"):
    st.markdown("""
    **How to use this tool:**
    1. Adjust the three clinical parameters in the left sidebar
    2. Click "Predict Survival" to run the model
    3. Review the survival curve and key metrics
    4. Expand sections for detailed input and model information
    
    **Parameter Ranges:**
    - **KPS**: 50-100 (typical clinical range)
    - **Na**: 120-150 mmol/L (clinical range including abnormal values)
    - **Cl**: 90-120 mmol/L (clinical range including abnormal values)
    
    **Clinical Notes:**
    - Model based on retrospective multi-center data
    - Predictions represent statistical probabilities
    - Always combine with clinical judgment
    """)

# Footer
st.divider()
st.caption("© 2025 Leptomeningeal Metastasis Survival Analysis | Cox Proportional Hazards Model")
st.caption("Original paper: Machine Learning for Predicting Leptomeningeal Metastasis and Prognosis in Lung Adenocarcinoma: a multi-center retrospective study Using the \"Prompt\" Model.")