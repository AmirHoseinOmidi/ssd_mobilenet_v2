import socket
import cv2
import struct
import numpy as np

# Client class for object detection
class ObjectDetectionClient:
    def __init__(self, server_ip='127.0.0.1', server_port=9999):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cap = None

    # Connect to the server
    def connect(self):
        print("[CLIENT] Connecting to server...")
        try:
            self.client_socket.connect((self.server_ip, self.server_port))
            print("[CLIENT] Connected successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    # Start webcam
    def start_camera(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return False
        return True

    # Send frames and receive results
    def send_frames(self):
        try:
            while True:
                # Read a frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break

                # Encode frame to JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_data = jpeg.tobytes()

                # Send frame length + data
                try:
                    self.client_socket.sendall(struct.pack(">L", len(frame_data)) + frame_data)
                except Exception as e:
                    print(f"[ERROR] Failed to send frame: {e}")
                    break

                # Receive processed image length + data
                try:
                    length_bytes = self.client_socket.recv(4)
                    if not length_bytes:
                        break
                    length = struct.unpack(">L", length_bytes)[0]

                    data = b''
                    while len(data) < length:
                        packet = self.client_socket.recv(min(length - len(data), 4096))
                        if not packet:
                            break
                        data += packet

                    if len(data) == length:
                        # Decode and display processed image
                        nparr = np.frombuffer(data, np.uint8)
                        processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        cv2.imshow('Processed Frame', processed_frame)
                    else:
                        print("[ERROR] Incomplete response from server")

                except Exception as e:
                    print(f"[ERROR] Failed to receive response: {e}")
                    break

                # Press ESC to exit
                if cv2.waitKey(1) == 27:
                    break

        finally:
            self.cleanup()

    
    # Cleanup and release resources
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.client_socket.close()
        print("[CLIENT] Connection closed.")

# Main function to run client
if __name__ == "__main__":
    client = ObjectDetectionClient()
    if client.connect() and client.start_camera():
        client.send_frames()
