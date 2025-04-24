import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from cotton_disease_detection import CottonDiseaseClassifier, get_severity
import os

def load_model():
    """Load the trained model"""
    try:
        checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
        model = CottonDiseaseClassifier(num_classes=len(checkpoint['class_names']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint['class_names']
    except FileNotFoundError:
        st.error("Model file 'best_model.pth' not found. Please train the model first.")
        return None, None

def predict_image(model, image, class_names):
    """Make prediction on uploaded image"""
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    return class_names[predicted], confidence.item()

def main():
    st.set_page_config(page_title="Cotton Disease Detection", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 class='title'>Cotton Disease Detection System</h1>", unsafe_allow_html=True)
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        return
    
    # File uploader
    st.write("### Upload an image of a cotton plant leaf")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        predicted_class, confidence = predict_image(model, image, class_names)
        severity = get_severity(confidence)
        
        # Display results
        with col2:
            st.write("### Results")
            
            # Disease prediction
            st.write("**Detected Disease:**")
            st.info(predicted_class)
            
            # # Confidence
            # st.write("**Confidence:**")
            # st.progress(confidence)
            # st.write(f"{confidence:.2%}")
            
            # Severity
            if predicted_class != "Healthy":
                st.write("**Disease Severity:**")
                if severity == "Mild":
                    st.success(f"🟢 {severity}")
                elif severity == "Moderate":
                    st.warning(f"🟡 {severity}")
                else:
                    st.error(f"🔴 {severity}")
            
            # Recommendations
            st.write("### Recommendations")
            recommendations = {
                "Healthy": "Continue regular maintenance and monitoring.",
                "Bacterial Blight": "1. Remove infected plants\n2. Apply copper-based bactericides\n3. Improve air circulation",
                "Curl Virus": "1. Control whitefly population\n2. Remove infected plants\n3. Use resistant varieties",
                "Fussarium Wilt": "1. Crop rotation\n2. Use resistant varieties\n3. Soil solarization",
                "Target spot": "1. Apply fungicides\n2. Improve drainage\n3. Reduce leaf wetness",
                "Aphids": "1. Use insecticidal soaps\n2. Introduce natural predators\n3. Remove infected parts",
                "Army worm": "1. Apply appropriate insecticides\n2. Monitor field regularly\n3. Use pheromone traps",
                "Powdery Mildew": "1. Apply fungicides\n2. Improve air circulation\n3. Avoid overhead irrigation"
            }
            
            if predicted_class in recommendations:
                st.info(recommendations[predicted_class])
            
    # Add information about the model
    st.sidebar.write("### About")
    st.sidebar.write("""
    This application uses a deep learning model based on ResNet50 architecture to detect diseases in cotton plants.
    
    The model can identify the following conditions:
    """)
    for class_name in class_names:
        st.sidebar.write(f"- {class_name}")
    
    st.sidebar.write("\n### Severity Levels")
    st.sidebar.write("🟢 **Mild** (Confidence < 50%)")
    st.sidebar.write("🟡 **Moderate** (Confidence 50-80%)")
    st.sidebar.write("🔴 **Severe** (Confidence > 80%)")

if __name__ == "__main__":
    main() 
