from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Input video
video_path = "datasets/challenge.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
output_path = "outputs/final_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Vehicle detection
    results = model(frame)

    detections = []

    for result in results[0].boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = result

        detections.append(
            ([x1, y1, x2 - x1, y2 - y1], score, class_id)
        )

    # Vehicle tracking
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Vehicle ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Lane detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)

    height_roi, width_roi = edges.shape

    polygon = np.array([[
        (0, height_roi),
        (width_roi, height_roi),
        (width_roi // 2, int(height_roi * 0.7))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        cropped_edges,
        2,
        np.pi / 180,
        100,
        minLineLength=80,
        maxLineGap=50
    )

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Remove noisy lines
            if abs(slope) < 0.5 or abs(slope) > 2:
                continue

            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    # Draw left lane
    for line in left_lines[:2]:

        x1, y1, x2, y2 = line[0]

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            6
        )

    # Draw right lane
    for line in right_lines[:2]:

        x1, y1, x2, y2 = line[0]

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            6
        )

    # Collision warning
    for result in results[0].boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = result

        box_height = y2 - y1

        if box_height > 300:

            cv2.putText(
                frame,
                "WARNING: VEHICLE TOO CLOSE",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    cv2.imshow("Autonomous Vehicle System", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.waitKey(1000)

out.release()

cv2.destroyAllWindows()

print("FINAL AUTONOMOUS SYSTEM COMPLETED")
