import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Sustainable Agriculture AI",
    page_icon="🌱",
    layout="wide"
)

# Define functions for data processing and model building
@st.cache_data
def load_data():
    """Load and preprocess the crop recommendation dataset"""
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure the file is in the correct location.")
        return None

def preprocess_data(df):
    """Preprocess the dataset for model training"""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        st.warning("Dropped rows with missing values")
    
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Convert categorical labels to numerical if needed
    if processed_df['label'].dtype == 'object':
        unique_crops = processed_df['label'].unique()
        crop_dict = {crop: i for i, crop in enumerate(unique_crops)}
        processed_df['crop_num'] = processed_df['label'].map(crop_dict)
    
    return processed_df

def train_model(df):
    """Train a Random Forest model on the dataset"""
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
        'accuracy': accuracy,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model_data

def predict_crop(model_data, input_values):
    """Predict the most suitable crop based on input soil and weather conditions"""
    # Extract the model and scaler
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Scale the input values
    input_scaled = scaler.transform([input_values])
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Get top 3 crop recommendations with their probabilities
    crop_names = model.classes_
    crop_probs = list(zip(crop_names, probabilities))
    top_crops = sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]
    
    return prediction[0], top_crops

def get_sustainability_score(input_values, crop):
    """Calculate a sustainability score based on how well the conditions match the crop's ideal conditions"""
    # Define optimal ranges for different crops (simplified)
    crop_optimal_conditions = {
        'rice': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'temperature': (22, 28), 'humidity': (80, 85), 'ph': (5.5, 6.5), 'rainfall': (200, 300)},
        'wheat': {'N': (100, 140), 'P': (50, 70), 'K': (40, 70), 'temperature': (15, 22), 'humidity': (60, 70), 'ph': (6.0, 7.0), 'rainfall': (75, 110)},
        'maize': {'N': (120, 160), 'P': (60, 80), 'K': (80, 100), 'temperature': (20, 30), 'humidity': (50, 65), 'ph': (5.8, 7.0), 'rainfall': (80, 110)},
        # Add more crops as needed
    }
    
    # Default to moderate conditions if the crop isn't in our database
    default_conditions = {'N': (80, 140), 'P': (40, 70), 'K': (40, 80), 'temperature': (15, 30), 'humidity': (50, 85), 'ph': (5.5, 7.0), 'rainfall': (75, 300)}
    
    # Get the optimal conditions for the crop or use default
    optimal = crop_optimal_conditions.get(crop.lower(), default_conditions)
    
    # Calculate how close each parameter is to the optimal range
    score = 0
    parameters = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    for i, param in enumerate(parameters):
        val = input_values[i]
        min_val, max_val = optimal[param]
        
        # If value is within optimal range, score is 1
        if min_val <= val <= max_val:
            score += 1
        else:
            # Calculate distance from optimal range (normalized)
            if val < min_val:
                distance = (min_val - val) / min_val
            else:
                distance = (val - max_val) / max_val
            
            # Score decreases as distance increases
            score += max(0, 1 - distance)
    
    # Normalize score to be between 0 and 100
    return (score / len(parameters)) * 100

def get_resource_optimization_tips(input_values, predicted_crop):
    """Generate resource optimization tips based on the input values and predicted crop"""
    # Extract input values
    n, p, k, temp, humidity, ph, rainfall = input_values
    
    tips = []
    
    # N-P-K related tips
    if n > 140:
        tips.append("Nitrogen levels are high. Consider reducing nitrogen fertilizer application to prevent runoff and groundwater contamination.")
    elif n < 80:
        tips.append("Nitrogen levels are low. Consider organic nitrogen sources like compost or legume cover crops.")
    
    if p > 70:
        tips.append("Phosphorus levels are high. Reduce phosphorus application to prevent water pollution.")
    elif p < 40:
        tips.append("Phosphorus levels are low. Consider bone meal or rock phosphate as organic alternatives.")
    
    if k > 80:
        tips.append("Potassium levels are high. Monitor and adjust potassium fertilizer application.")
    elif k < 40:
        tips.append("Potassium levels are low. Consider wood ash or seaweed as natural potassium sources.")
    
    # pH related tips
    if ph < 5.5:
        tips.append("Soil is acidic. Consider adding agricultural lime to raise pH.")
    elif ph > 7.5:
        tips.append("Soil is alkaline. Consider adding organic matter like compost to lower pH gradually.")
    
    # Rainfall related tips
    if rainfall < 80:
        tips.append("Low rainfall area. Consider drip irrigation or mulching to conserve water.")
    elif rainfall > 200:
        tips.append("High rainfall area. Consider raised beds or improving drainage to prevent waterlogging.")
    
    # Temperature and humidity related tips
    if temp > 30:
        tips.append("Temperature is high. Consider shade cloth or intercropping with taller plants to provide shade.")
    elif temp < 15:
        tips.append("Temperature is low. Consider cold frames or row covers to protect crops.")
    
    if humidity > 80:
        tips.append("Humidity is high. Increase plant spacing to improve air circulation and reduce fungal diseases.")
    elif humidity < 50:
        tips.append("Humidity is low. Consider mulching to conserve soil moisture.")
    
    # Crop-specific tips
    if predicted_crop.lower() == 'rice':
        tips.append("For rice cultivation, consider using the System of Rice Intensification (SRI) to reduce water usage by up to 50%.")
    elif predicted_crop.lower() == 'wheat':
        tips.append("For wheat, consider conservation tillage to reduce soil erosion and improve soil health.")
    elif predicted_crop.lower() == 'maize':
        tips.append("For maize, consider intercropping with legumes to reduce nitrogen fertilizer needs.")
    
    # General sustainable farming tips
    tips.append("Consider crop rotation to break pest cycles and improve soil health.")
    tips.append("Implement integrated pest management (IPM) to reduce pesticide use.")
    tips.append("Plant cover crops during off-seasons to prevent soil erosion and add organic matter.")
    
    return tips

# Main application
def main():
    st.title("🌱 Sustainable Agriculture AI")
    st.markdown("""
    This application helps farmers make data-driven decisions to increase crop yields while reducing environmental impact.
    Upload your soil and weather data to get crop recommendations and sustainability tips.
    """)
    
    # Create tabs for different functionality
    tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Crop Prediction", "Sustainability Metrics", "About"])
    
    # Load and display the dataset
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Data Analysis Tab
    with tab1:
        st.header("Dataset Exploration")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(df.head())
            st.write(f"Dataset shape: {df.shape}")
            
            # Display descriptive statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
            # Count of each crop type
            st.subheader("Crop Distribution")
            crop_counts = df['label'].value_counts().reset_index()
            crop_counts.columns = ['Crop', 'Count']
            st.dataframe(crop_counts)
        
        with col2:
            # Create visualizations
            st.subheader("Visualizations")
            
            # Crop distribution pie chart
            fig1 = px.pie(crop_counts, values='Count', names='Crop', title='Crop Distribution in Dataset')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Between Features")
            corr = df.select_dtypes(include=[np.number]).corr()
            fig2 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            fig2.update_layout(title='Feature Correlation Heatmap')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Feature distribution by crop
            feature_to_plot = st.selectbox(
                "Select a feature to visualize distribution by crop:",
                ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            )
            
            fig3 = px.box(df, x='label', y=feature_to_plot, title=f'{feature_to_plot} Distribution by Crop')
            fig3.update_layout(xaxis_title='Crop', yaxis_title=feature_to_plot)
            st.plotly_chart(fig3, use_container_width=True)
    
    # Train the model
    processed_df = preprocess_data(df)
    model_data = train_model(processed_df)
    
    # Crop Prediction Tab
    with tab2:
        st.header("Crop Recommendation System")
        st.write(f"Model Accuracy: {model_data['accuracy']:.2%}")
        
        # Create two columns for input and output
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Enter Soil and Climate Conditions")
            
            # Input form for soil and weather parameters
            n_value = st.slider("Nitrogen (N) - kg/ha", 0, 140, 50)
            p_value = st.slider("Phosphorus (P) - kg/ha", 0, 145, 50)
            k_value = st.slider("Potassium (K) - kg/ha", 0, 205, 50)
            temp_value = st.slider("Temperature (°C)", 0.0, 45.0, 25.0)
            humidity_value = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
            ph_value = st.slider("pH", 0.0, 14.0, 6.5)
            rainfall_value = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
            
            input_values = [n_value, p_value, k_value, temp_value, humidity_value, ph_value, rainfall_value]
            
            # Button to predict
            if st.button("Predict Suitable Crop"):
                # Make prediction
                prediction, top_crops = predict_crop(model_data, input_values)
                
                # Store prediction in session state for other tabs
                st.session_state.prediction = prediction
                st.session_state.input_values = input_values
        
        with col2:
            # Display prediction results
            if 'prediction' in st.session_state:
                st.subheader("Prediction Results")
                st.success(f"The most suitable crop is: **{st.session_state.prediction}**")
                
                # Display top 3 recommendations with probabilities
                st.subheader("Top Recommendations")
                for crop, prob in top_crops:
                    st.write(f"{crop}: {prob:.2%} confidence")
                
                # Feature importance
                st.subheader("Feature Importance")
                
                # Get feature importance from model
                importances = model_data['model'].feature_importances_
                features = model_data['features']
                
                # Create a bar chart
                importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Feature', y='Importance', 
                             title='Feature Importance for Crop Recommendation')
                st.plotly_chart(fig, use_container_width=True)
                
                # Confusion matrix
                st.subheader("Model Performance")
                
                # Calculate and display the confusion matrix
                cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                classes = model_data['model'].classes_
                
                fig = px.imshow(cm, 
                                x=classes,
                                y=classes,
                                labels=dict(x="Predicted Crop", y="Actual Crop", color="Count"),
                                title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
    
    # Sustainability Metrics Tab
    with tab3:
        st.header("Sustainability Analysis and Recommendations")
        
        if 'prediction' in st.session_state:
            # Extract the stored prediction and input values
            prediction = st.session_state.prediction
            input_values = st.session_state.input_values
            
            # Calculate sustainability score
            score = get_sustainability_score(input_values, prediction)
            
            # Display sustainability score with a gauge chart
            st.subheader("Sustainability Score")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sustainability Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate and display resource optimization tips
            st.subheader("Resource Optimization Recommendations")
            
            tips = get_resource_optimization_tips(input_values, prediction)
            
            for i, tip in enumerate(tips, 1):
                st.write(f"{i}. {tip}")
            
            # Display radar chart to show how the current conditions compare to optimal conditions
            st.subheader("Current vs. Optimal Conditions")
            
            # Simplified optimal conditions for the predicted crop
            optimal_conditions = {
                'rice': [100, 50, 50, 25, 82, 6.0, 250],
                'wheat': [120, 60, 60, 18, 65, 6.5, 95],
                'maize': [140, 70, 90, 25, 60, 6.5, 95],
                # Default for other crops
            }.get(prediction.lower(), [110, 55, 55, 22, 70, 6.5, 120])
            
            # Normalize the values for radar chart
            max_values = [140, 145, 205, 45, 100, 14, 300]  # Maximum possible values
            
            normalized_input = [min(100, (i/m)*100) for i, m in zip(input_values, max_values)]
            normalized_optimal = [min(100, (o/m)*100) for o, m in zip(optimal_conditions, max_values)]
            
            # Create radar chart
            categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_input,
                theta=categories,
                fill='toself',
                name='Current Conditions'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_optimal,
                theta=categories,
                fill='toself',
                name='Optimal Conditions'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Current vs. Optimal Conditions for " + prediction
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Environmental impact estimation
            st.subheader("Environmental Impact Estimation")
            
            # Simplified environmental impact calculation
            # Higher values mean higher environmental impact
            n_impact = abs(input_values[0] - optimal_conditions[0]) / max_values[0] * 10
            p_impact = abs(input_values[1] - optimal_conditions[1]) / max_values[1] * 10
            k_impact = abs(input_values[2] - optimal_conditions[2]) / max_values[2] * 10
            water_impact = abs(input_values[6] - optimal_conditions[6]) / max_values[6] * 10
            
            # Create impact metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(label="Nitrogen Runoff Risk", value=f"{n_impact:.1f}/10")
                
            with col2:
                st.metric(label="Phosphorus Leaching Risk", value=f"{p_impact:.1f}/10")
                
            with col3:
                st.metric(label="Potassium Balance", value=f"{k_impact:.1f}/10")
                
            with col4:
                st.metric(label="Water Usage Efficiency", value=f"{water_impact:.1f}/10")
        else:
            st.info("Please go to the Crop Prediction tab and make a prediction first.")
    
    # About Tab
    with tab4:
        st.header("About This Project")
        st.markdown("""
        ### Sustainable Agriculture with AI for Crop Yield Prediction
        
        This application demonstrates how AI can help farmers increase crop yields while reducing environmental impact. 
        By analyzing soil conditions and climate data, our model provides recommendations for the most suitable crops 
        and offers suggestions to optimize resource usage.
        
        #### Features:
        - **Crop Recommendation**: Predicts the most suitable crop based on soil and climate conditions
        - **Data Analysis**: Explores the relationships between environmental factors and crop suitability
        - **Sustainability Metrics**: Provides a sustainability score and resource optimization recommendations
        - **Environmental Impact**: Estimates the potential environmental impact of farming decisions
        
        #### How This Addresses Global Food Security:
        - Increases crop yields by matching crops to optimal growing conditions
        - Reduces resource waste by providing targeted recommendations
        - Promotes sustainable farming practices that preserve soil health
        - Makes agriculture more resilient to climate change by adapting crop choices
        
        #### Developing Country Applications:
        - Low-tech interface that can work on basic smartphones
        - Recommendations that consider resource constraints
        - Focus on sustainable, low-input farming methods
        - Support for traditional crops that are well-adapted to local conditions
        
        #### Data Source:
        This application uses the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle, 
        which contains information about soil conditions, climate factors, and suitable crops.
        """)

if __name__ == "__main__":
    main()