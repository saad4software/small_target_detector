from ultralytics import YOLO
import cv2
import time

model_path = "train/weights/last.pt"
model = YOLO(model_path)


file_name="DRONE_113"
cap = cv2.VideoCapture(f"../Data/Video_V/V_{file_name}.mp4")
# cap = cv2.VideoCapture(f"../Data/Video_IR/IR_{file_name}.mp4")

counter = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames

        start = time.time()

        # frame = cv2.resize(frame, (320, 256), interpolation = cv2.INTER_LINEAR)
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # cv2.imwrite(f"output_images/{file_name}_{counter}.png", annotated_frame)
        counter += 1
        
        end = time.time()
        print(f"elapsed time {end - start}")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

