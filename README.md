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

⚙️ Running the Code
To run the real-time object detection, follow these steps:

Ensure you have Python 3 and the necessary libraries installed (tensorflow, opencv-python, numpy).

Download the pre-trained SSD MobileNetV2 model from the official TensorFlow model zoo:

Pretrained Model: SSD MobileNetV2 - COCO17

Extract the model file and point to the directory where it is stored in the code (model_dir).


The webcam will start capturing and performing object detection in real-time. Press 'q' to exit.

📚 References
MobileNetV2: For details on MobileNetV2, please refer to the original paper.

Pretrained Model: The pretrained model used in this project was obtained from the TensorFlow Object Detection Model Zoo.

🎥 Demo Video:

Watch it in action: Demo Video

📊 Output Example When you quit, you'll see a summary in the terminal:

Detection Finished. Results: person: 0.810
person: 0.591
person: 0.588
