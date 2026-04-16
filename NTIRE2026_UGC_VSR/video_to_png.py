import cv2
import os

video_dir = "./Video/videos"
output_base = "./Video/LQ_images"

video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

for filename in sorted(os.listdir(video_dir)):
    if not filename.lower().endswith(video_extensions):
        continue

    video_path = os.path.join(video_dir, filename)
    video_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(output_base, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{idx:010d}.png"), frame)
        idx += 1
    cap.release()
    print(f"[{filename}] Done! Total frames: {idx - 1}")

print("All videos processed!")