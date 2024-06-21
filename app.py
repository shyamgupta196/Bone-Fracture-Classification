import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch.nn as nn 

# Load the pre-trained model
model = EfficientNet.from_name('efficientnet-b6')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification

model.load_state_dict(torch.load('best-model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size expected by the model
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Streamlit app
st.title('Bone Fracture Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert("RGB")  # Ensure the image is in RGB format
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Transform the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    # Display the prediction
    class_names = ['Not Fractured', 'Fractured']  # Replace with your actual class names
    print(output.data)
    print(predicted)
    st.write(f'Predicted Class: {class_names[predicted.item()]}')

# Uncomment the following line if running this script directly
# if __name__ == "__main__":
#     st.run()
