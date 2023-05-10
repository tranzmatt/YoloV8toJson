import argparse
import cv2
import numpy as np
import torch
import json

# Start a loop to read frames from the video stream
def yolo_video_processing(input_source=None, model=None, confidence=0.5,
                          display_video=False, print_json=False):
    # Load the YOLO model
    weights = model + '.pt'
    model = torch.hub.load('ultralytics/yolov8', model, weights, conf_thres=confidence)

    # Create a video capture object
    cap = cv2.VideoCapture(input_source)

    # Get the frame width and height
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a canvas for drawing the detected objects
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If the frame was not read successfully, break out of the loop
        if not ret:
            break

        # Convert the frame to a tensor
        frame = torch.from_numpy(frame).float()

        # Run the YOLO model on the frame
        detections = model(frame)

        # Get the bounding boxes and confidence levels of the detected objects
        boxes = detections[0]['boxes']
        confidences = detections[0]['confidences']

        # Draw the detected objects on the canvas
        for box, confidence in zip(boxes, confidences):
            # Get the object type
            object_type = detections[0]['names'][int(box[5])]

            # Get the bounding box coordinates
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))

            # Draw the bounding box on the canvas
            cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 2)

            # Write the object type, confidence level, and bounding box coordinates to the canvas
            cv2.putText(canvas, f'{object_type} ({confidence:.2f}) ({top_left[0]} {top_left[1]} {bottom_right[0]} {bottom_right[1]})', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Display the canvas
        if display_video:
            cv2.imshow('Detected Objects', canvas)
            cv2.waitKey(1)

        # Write the detected objects to a JSON file
        if print_json:
            with open('detected_objects.json', 'w') as f:
                json.dump(detections, f, indent=4)

        # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


# Call the yolo_video_processing function
if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('-i', '--input', type=str, help='The input source.')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='The confidence threshold.')
    parser.add_argument('-s', '--display_video', action='store_true', help='Display the object detections.')
    parser.add_argument('-p', '--print_json', action='store_true', help='Print the JSON to the console.')
    parser.add_argument('-y', '--model', type=str, default='yolov8s', help='The desired YOLO model to use.')

    # Parse the arguments
    args = parser.parse_args()

    yolo_video_processing(args.input, args.model, args.confidence, args.display_video, args.print_json)


