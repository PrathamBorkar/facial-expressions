import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Define your model class
class facialexpressiondataset(nn.Module):
    def __init__(self):
        super(facialexpressiondataset, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),  # Output size is 8 for 8 emotion classes
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = facialexpressiondataset().to(device)

# Load the model's state dict (weights)
model.load_state_dict(torch.load(r"c:\Users\Pratham Borkar\Desktop\facial_rec_model\facial_expression_model.pth", map_location=device))
model.eval()  # Set the model to evaluation mode after loading the weights

# Emotion mapping
emotion_mapping = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Neutral',
    7: 'Contempt'
}

# Preprocess function
def preprocess_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Resize to 48x48
    resized_frame = cv2.resize(blurred_frame, (48, 48))
    
    # Normalize and convert to tensor
    img_array = np.array(resized_frame, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    return img_tensor

# Real-time video capture and prediction
def main():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the application.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        input_tensor = preprocess_frame(frame)

        # Predict emotion
        with torch.no_grad():
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_emotion = emotion_mapping[predicted_label]

        # Display the emotion on the frame
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Real-Time Emotion Recognition', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
