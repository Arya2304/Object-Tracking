import numpy as np
import time
import collections
try:
    import cv2 
    from cv2 import (
        imread, selectROI, destroyWindow, cvtColor, COLOR_BGR2HSV,
        calcHist, normalize, NORM_MINMAX, VideoCapture, CAP_PROP_FPS,
        TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, calcBackProject,
        meanShift, rectangle, putText, FONT_HERSHEY_SIMPLEX,
        imshow, waitKey, destroyAllWindows, line, circle
    )
except ImportError:
    raise ImportError("OpenCV (cv2) not found. Please install with: pip install opencv-python")

def select_roi(img):
    """Let user manually select ROI if automatic detection fails"""
    print("Please select the book cover region and press ENTER")
    roi = selectROI("Select Book Cover", img, False)
    destroyWindow("Select Book Cover")
    return tuple(roi) if all(roi) else (354, 252, 101, 143)  # Ensure ROI is a tuple

# Visualization settings
HISTORY_LENGTH = 100  # Number of frames to track
VIZ_WIDTH = 400       # Visualization window width
VIZ_HEIGHT = 300      # Visualization window height
MARGIN = 20           # Margin around plots

print(type(VIZ_HEIGHT))
print(type(VIZ_WIDTH))

def create_viz_window():
    """Create blank visualization window with fixed grid"""
    try:
        # Attempt to create a blank visualization window
        viz = np.zeros((VIZ_HEIGHT, VIZ_WIDTH, 3), dtype=np.uint8)  # Use np.zeros and np.uint8 explicitly
    except AttributeError as e:
        raise AttributeError("Error using numpy.zeros. Ensure numpy is installed and imported correctly.") from e

    # Draw fixed grid
    color = (50, 50, 50)
    # Vertical grid lines
    for x in range(0, VIZ_WIDTH, 50):
        line(viz, (x, 0), (x, VIZ_HEIGHT), color, 1)
    # Horizontal grid lines
    for y in range(0, VIZ_HEIGHT, 50):
        line(viz, (0, y), (VIZ_WIDTH, y), color, 1)
    
    return viz


def update_viz_scatter(viz, history):
    """Update the scatter plot visualization with axes and labels"""
    # Clear the visualization window
    viz[:, :] = 0

    # Draw X and Y axes
    line(viz, (MARGIN, 0), (MARGIN, VIZ_HEIGHT), (255, 255, 255), 2)  # Y-axis
    line(viz, (0, VIZ_HEIGHT - MARGIN), (VIZ_WIDTH, VIZ_HEIGHT - MARGIN), (255, 255, 255), 2)  # X-axis

    # Add axis labels
    putText(viz, "X", (VIZ_WIDTH - 30, VIZ_HEIGHT - 10), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    putText(viz, "Y", (10, 20), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if len(history) > 1:
        # Calculate scaling factors
        max_x = max(h[0] for h in history)
        min_x = min(h[0] for h in history)
        max_y = max(h[1] for h in history)
        min_y = min(h[1] for h in history)

        x_scale = (VIZ_WIDTH - 2 * MARGIN) / max(1, max_x - min_x)
        y_scale = (VIZ_HEIGHT - 2 * MARGIN) / max(1, max_y - min_y)

        # Plot scatter points
        for (x, y, w, h) in history:
            px = MARGIN + int((x - min_x) * x_scale)
            py = VIZ_HEIGHT - MARGIN - int((y - min_y) * y_scale)
            circle(viz, (px, py), 3, (255, 255, 0), -1)  # Yellow scatter points

    # Add a label to the scatter plot
    putText(viz, "Scatter Plot", (10, 25), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return viz


def update_viz_tracking_history(viz, fps_history, accuracy_history):
    """Update the tracking history visualization with FPS and accuracy line plots"""
    # Clear the visualization window
    viz[:, :] = 0

    # Draw X and Y axes
    line(viz, (MARGIN, 0), (MARGIN, VIZ_HEIGHT), (255, 255, 255), 2)  # Y-axis
    line(viz, (0, VIZ_HEIGHT - MARGIN), (VIZ_WIDTH, VIZ_HEIGHT - MARGIN), (255, 255, 255), 2)  # X-axis

    # Add axis labels
    putText(viz, "Time", (VIZ_WIDTH - 50, VIZ_HEIGHT - 10), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    putText(viz, "Value", (10, 20), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Plot FPS and accuracy
    if len(fps_history) > 1 and len(accuracy_history) > 1:
        # Scale FPS and accuracy to fit within the window
        max_fps = max(fps_history)
        max_accuracy = max(accuracy_history)
        fps_scale = (VIZ_HEIGHT - 2 * MARGIN) / max(1, max_fps)
        accuracy_scale = (VIZ_HEIGHT - 2 * MARGIN) / max(1, max_accuracy)

        # Plot FPS
        prev_fps = None
        for i, fps in enumerate(fps_history):
            px = MARGIN + int(i * (VIZ_WIDTH - 2 * MARGIN) / len(fps_history))
            py = VIZ_HEIGHT - MARGIN - int(fps * fps_scale)
            if prev_fps:
                line(viz, prev_fps, (px, py), (0, 255, 0), 2)  # Green line for FPS
            prev_fps = (px, py)

        # Plot accuracy
        prev_accuracy = None
        for i, accuracy in enumerate(accuracy_history):
            px = MARGIN + int(i * (VIZ_WIDTH - 2 * MARGIN) / len(accuracy_history))
            py = VIZ_HEIGHT - MARGIN - int(accuracy * accuracy_scale)
            if prev_accuracy:
                line(viz, prev_accuracy, (px, py), (255, 0, 0), 2)  # Red line for accuracy
            prev_accuracy = (px, py)

    # Add a label to the tracking history
    putText(viz, "Tracking History (FPS & Accuracy)", (10, 25), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return viz


def main():
    try:
        # Initialize tracking history
        history = collections.deque(maxlen=HISTORY_LENGTH)
        fps_history = collections.deque(maxlen=HISTORY_LENGTH)
        accuracy_history = collections.deque(maxlen=HISTORY_LENGTH)
        successful_frames = 0  # Counter for successful tracking frames
        
        # Create visualization windows
        viz_scatter = create_viz_window()
        viz_history = create_viz_window()
        
        # Load book cover image
        img = imread('./book_img/book_cover.jpg')
        if img is None:
            raise FileNotFoundError("Could not load book_img/book_cover.jpg")
        
        # Let user select ROI or use default
        x, y, width, height = select_roi(img)
        roi = img[y:y+height, x:x+width]
        
        # Convert ROI to HSV and calculate histogram
        hsv_roi = cvtColor(roi, COLOR_BGR2HSV)
        roi_hist = calcHist([hsv_roi], [0], None, [180], [0, 180])
        normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX)
        
        # Setup video capture
        cap = VideoCapture(0)
        if not cap.isOpened():
            print("Video capture device is not available.")
            raise IOError("Cannot open video capture")
            
        fps = cap.get(CAP_PROP_FPS)
        term_crit = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 20, 1)  # Increased iterations
        track_window = (x, y, width, height)
        
        print("Tracking started. Press ESC to exit")
        start_time = time.time()
        frame_count = 0
        total_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            total_fps += fps
            
            # Convert frame to HSV
            hsv = cvtColor(frame, COLOR_BGR2HSV)
            
            # Calculate back projection
            dst = calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            # Apply meanshift
            tracking_ok, track_window = meanShift(dst, track_window, term_crit)
            
            if tracking_ok:
                # Draw tracking rectangle and info
                x, y, w, h = track_window
                rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                putText(frame, "Arya's Book", (x, y-10), FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Update tracking history
                history.append((x, y, w, h))
                tracking_status = "Tracking Active"
                successful_frames += 1
            else:
                # Clear history when tracking is lost
                history.clear()
                tracking_status = "Tracking Lost"
            
            # Calculate accuracy
            accuracy = (successful_frames / frame_count) * 100 if frame_count > 0 else 0
            fps_history.append(fps)
            accuracy_history.append(accuracy)
            
            # Display debug info
            putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            putText(frame, f"Frame: {frame_count}", (10, 60), 
                    FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            putText(frame, f"Status: {tracking_status}", (10, 90), 
                    FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if tracking_status == "Tracking Lost" else (0,255,0), 2)
            if tracking_ok:
                putText(frame, f"Coords: ({x}, {y}, {w}, {h})", (10, 120), 
                        FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # Update and display visualizations
            viz_scatter = update_viz_scatter(viz_scatter, history)
            viz_history = update_viz_tracking_history(viz_history, fps_history, accuracy_history)
            
            # Display results in separate windows
            imshow('Tracking', frame)
            imshow('Back Projection', dst)
            imshow('Scatter Plot', viz_scatter)
            imshow('Tracking History', viz_history)
            
            # Exit on ESC
            key = waitKey(1) & 0xFF
            if key == 27:
                break
                
        # Calculate average FPS
        avg_fps = total_fps / frame_count if frame_count > 0 else 0
        print(f"\nTracking session ended.")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nReleasing resources...")
        if 'cap' in locals() and cap is not None:
            cap.release()
        destroyAllWindows()

if __name__ == "__main__":
    main()
