import cv2

# Open video capture source (webcam)
# cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera index

# Check if the camera was successfully opened


# Open video capture source with DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   

if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# Read and display video frames until the user presses 'q' key
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Failed to read a frame from the camera")
        break

    # Display the frame in a window named 'Video'
    cv2.imshow("Video", frame)

    # Wait for the user to press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
