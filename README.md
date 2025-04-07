# Real-Time Object Detection with SSD MobileNetV2

This project uses **TensorFlow** and **OpenCV** to perform real-time object detection from a webcam using the lightweight **SSD MobileNetV2** model.

## 🔍 What It Does
- Captures video from your webcam
- Detects over 90 object types (like people, cars, dogs)
- Draws bounding boxes with class names and confidence scores
- Filters out detections with confidence scores below 0.5
- Limits frame rate to 15 FPS for smooth performance

## 🧠 Tech Used
- **TensorFlow** – Loads and runs the SSD MobileNetV2 model
- **OpenCV** – Handles webcam input and drawing boxes
- **NumPy** – Works with image data
- **Time** – Controls frame rate

## ▶️ How It Works
1. Captures frames from the webcam
2. Resizes each frame to **320x320**
3. Converts it to a tensor and runs it through the model
4. If objects are detected:
   - Draws bounding boxes
   - Displays class names and scores (if score > 0.5)
5. Press **'q'** to stop

## 📦 Installation

Install the required libraries:

```bash
pip install tensorflow opencv-python numpy
```
‍‍‍‍‍‍‍
🎥 Demo Video:

Watch it in action: (https://youtu.be/bPFM69b0MGg)

📊 Output Example
When you quit, you'll see a summary in the terminal:

{
  "person": {"count": 3, "average_confidence": 0.85},
  "car": {"count": 2, "average_confidence": 0.92},
  "dog": {"count": 1, "average_confidence": 0.89}
}
