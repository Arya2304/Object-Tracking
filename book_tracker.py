import cv2 as cv
import numpy as np

def main():
    try:
        # Load book cover image
        img = cv.imread('./book_cover.jpg')
        if img is None:
            raise FileNotFoundError("Could not load book_cover.jpg")
        
        # Set ROI coordinates
        x, y, width, height = 354, 252, 101, 143
        roi = img[y:y+height, x:x+width]
        
        # Convert ROI to HSV and calculate histogram
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        
        # Setup video capture
        cap = cv.VideoCapture(1)
        if not cap.isOpened():
            raise IOError("Cannot open video capture")
            
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        track_window = (x, y, width, height)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to HSV
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Calculate back projection
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            # Apply meanshift
            ret, track_window = cv.meanShift(dst, track_window, term_crit)
            
            # Draw tracking rectangle
            x, y, w, h = track_window
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display results
            cv.imshow('Tracking', frame)
            cv.imshow('Back Projection', dst)
            
            # Exit on ESC
            key = cv.waitKey(1) & 0xFF
            if key == 27:
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
