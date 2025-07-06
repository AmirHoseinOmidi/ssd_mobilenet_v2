# Real-Time Object Detection Client-Server System

This project implements a client-server architecture for real-time object detection using **SSD MobileNetV2** with **TensorFlow** and **OpenCV**.

🎥 Demo Video:

Watch it in action: https://youtu.be/bPFM69b0MGg


## 🔍 Project Structure

- `server.py`: Processes images using SSD MobileNetV2 model
- `client.py`: Captures webcam frames and displays results

## 🧠 Tech Stack

- **TensorFlow** - Object detection model
- **OpenCV** - Image processing and webcam capture
- **NumPy** - Array operations
- **Socket Programming** - Client-server communication

## ▶️ How It Works

### System Workflow
1. Client captures frames from webcam
2. Frames are compressed and sent to server
3. Server processes frames using TensorFlow model
4. Server draws bounding boxes and labels (confidence > 0.5)
5. Processed frames are sent back to client
6. Client displays the annotated frames

### Key Features
- Real-time detection of 91 COCO object categories
- Multi-threaded server handles multiple clients
- Efficient JPEG compression for network transmission
- Confidence threshold filtering (0.5 default)



### Requirements
```bash
pip install tensorflow opencv-python numpy
