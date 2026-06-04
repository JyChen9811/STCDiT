from PIL import Image
import cv2
import numpy as np
from typing import List
import imageio

def analyze_video_motion(video_path,
                                        minimal_segment_length=5,
                                        shi_tomasi_max_corners=100,
                                        shi_tomasi_quality_level=0.3,
                                        shi_tomasi_min_distance=7,
                                        lk_window_size=(21, 21),
                                        lk_max_level=3,
                                        lk_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                                        trans_threshold=5, 
                                        rot_threshold=0.5, 
                                        scale_threshold=0.01): 

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return



    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray,
                                          maxCorners=shi_tomasi_max_corners,
                                          qualityLevel=shi_tomasi_quality_level,
                                          minDistance=shi_tomasi_min_distance)

    if prev_points is None:
        print("No good features to track in the first frame. Exiting.")
        cap.release()
        return

    print("开始分析视频运动，并保存检测到抖动或缩放的帧...")

    frame_count = 0
    frame_filter_select = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                               current_gray,
                                                               prev_points,
                                                               None,
                                                               winSize=lk_window_size,
                                                               maxLevel=lk_max_level,
                                                               criteria=lk_criteria)

        
        if current_points is not None and prev_points is not None and status is not None:
            good_prev_points = prev_points[status == 1]
            good_current_points = current_points[status == 1]
        else:
            good_prev_points = np.array([])
            good_current_points = np.array([])

        motion_info_text = ""
        is_shake = False
        is_zoom = False

        
        if len(good_prev_points) >= 3 and len(good_current_points) >= 3:
            M, _ = cv2.estimateAffine2D(good_prev_points, good_current_points)

            if M is not None:
                dx = M[0, 2]
                dy = M[1, 2]
                angle_rad = np.arctan2(M[1, 0], M[0, 0])
                angle_deg = np.degrees(angle_rad)
                scale = (np.sqrt(M[0, 0]**2 + M[1, 0]**2) + np.sqrt(M[0, 1]**2 + M[1, 1]**2)) / 2.0

                if abs(dx) > trans_threshold or abs(dy) > trans_threshold:
                    is_shake = True
                    
                if abs(angle_deg) > rot_threshold:

                    is_shake = True
                    
                if abs(scale - 1.0) > scale_threshold:
                    is_zoom = True
                    
                motion_info_text = f"Frame {frame_count}: "
                if is_shake:
                    motion_info_text += f"SHAKE! (Trans: {dx:.2f},{dy:.2f}, Rot: {angle_deg:.2f}) "
                    frame_filter_select.append(frame_count)
                if is_zoom:
                    motion_info_text += f"ZOOM! (Scale: {scale:.3f}) "
                    frame_filter_select.append(frame_count)
                if not is_shake and not is_zoom:
                    motion_info_text += f"Stable (Trans: {dx:.2f},{dy:.2f}, Rot: {angle_deg:.2f}, Scale: {scale:.3f})"
                
                    

            else:
                motion_info_text = f"Frame {frame_count}: Could not compute affine transform."
                
        else:
            motion_info_text = f"Frame {frame_count}: Not enough matching points."
            
            prev_points = cv2.goodFeaturesToTrack(current_gray,
                                                  maxCorners=shi_tomasi_max_corners,
                                                  qualityLevel=shi_tomasi_quality_level,
                                                  minDistance=shi_tomasi_min_distance)
            if prev_points is None or len(prev_points) < 3:
                print(f"Frame {frame_count}: Still no sufficient features after re-detection. Skipping remaining frames.")
                break

            

        prev_gray = current_gray.copy()
        prev_points = cv2.goodFeaturesToTrack(prev_gray,
                                              maxCorners=shi_tomasi_max_corners,
                                              qualityLevel=shi_tomasi_quality_level,
                                              minDistance=shi_tomasi_min_distance)
        if prev_points is None:
            print(f"Frame {frame_count}: No features detected for next frame. Breaking.")
            break
    
    cap.release()

    if len(frame_filter_select) == 0:
        segments = []
        step = minimal_segment_length
        for start in range(0, frame_count, step):
            end = min(start + step, frame_count)  
            segments.append((start, end))
        segments[-1] = (segments[-1][0], segments[-1][1] + 1)
        return segments

    frame_filter_select = sorted(list(set(frame_filter_select)))

    cursor = 0
    segments = []
    frame_filter_select = [0] + frame_filter_select
    
    for i in range(len(frame_filter_select)-1):
        length = frame_filter_select[i+1] - frame_filter_select[i]
        cursor = cursor + length    

        if length > minimal_segment_length:
            start = frame_filter_select[i]
            print(f"start {start, frame_filter_select[i + 1]} ")
            while start + minimal_segment_length < frame_filter_select[i + 1]:
                segments.append((start, start + minimal_segment_length))
                start += minimal_segment_length
            # 补最后不足 minimal_segment_length 的一段
            if start < frame_filter_select[i + 1]:
                segments.append((start, frame_filter_select[i + 1]))
        else:
            segments.append((frame_filter_select[i], frame_filter_select[i + 1]))

    if cursor < frame_count+1:

        length = frame_count - frame_filter_select[-1]
        if length > minimal_segment_length:
            start = frame_filter_select[-1]
            while start + minimal_segment_length < frame_count+1:
                segments.append((start, start + minimal_segment_length))
                start += minimal_segment_length
            if start < frame_count+1:
                segments.append((start, frame_count+1))

        else:
            segments.append((cursor, frame_count+1))

    else:
        segments[-1] = (segments[-1][0], segments[-1][1] + 1)

    
    return segments