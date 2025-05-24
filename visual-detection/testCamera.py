import cv2

# Desired resolution
width = 1280
height = 720

# Open the first camera (index 0)
cap = cv2.VideoCapture(0)

# Set the desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Read a frame to apply the settings
ret, frame = cap.read()

# Get the actual resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")

# Display the video feed
while ret:
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

# Release resources
cap.release()
cv2.destroyAllWindows()
