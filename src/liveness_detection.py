import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE_LANDMARKS = [159, 145]  # eyelid points

def is_blinking(landmarks, image_height):
    top = landmarks[159].y * image_height
    bottom = landmarks[145].y * image_height
    return abs(top - bottom) < 4  # adjust threshold if needed

cap = cv2.VideoCapture(0)
blink_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            if is_blinking(landmarks.landmark, h):
                blink_count += 1
                cv2.putText(frame, "Blink Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Liveness Detection", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27 or blink_count >= 1:  # ESC or one blink = live
        break

cap.release()
cv2.destroyAllWindows()

if blink_count >= 1:
    print("✅ Liveness Confirmed (Blink Detected)")
else:
    print("❌ Spoofing Suspected (No Blink)")
