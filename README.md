# Object-Tracking

The project is a real-time object tracking application focused on tracking a book cover using a webcam feed. It leverages OpenCV and numpy libraries to perform color-based tracking with visualization of tracking performance.

How it works:

1. Initialization:

- The program loads a reference image of a book cover from the local directory.
- The user is prompted to select the region of interest (ROI) on the book cover image manually. If the user does not select, a default ROI is used.
- The selected ROI is converted to the HSV color space, and a color histogram of the ROI is calculated and normalized. This histogram serves as the color model for tracking.

2. Video Capture and Tracking:

- The program opens the default webcam for video capture.
- For each captured frame:
- The frame is converted to HSV color space.
- A back projection is computed using the ROI histogram, highlighting areas in the frame that match the color distribution of the book cover.
- The meanshift algorithm is applied to the back projection to find the new location of the book cover in the frame.
- If tracking is successful, a green rectangle is drawn around the detected book cover, and tracking information (coordinates, status) is displayed.
- If tracking is lost, the tracking history is cleared, and the status is updated accordingly.

3. Visualization:

- The program maintains a history of tracked positions, FPS, and tracking accuracy.
- It displays multiple windows:
- The live tracking video with annotations.
- The back projection image showing the color match.
- A scatter plot visualizing the tracked positions over time.
- A line plot showing tracking history with FPS and accuracy metrics.

4. User Interaction and Exit:

- The user can exit the tracking loop by pressing the ESC key.
- Upon exit, the program releases the webcam and closes all OpenCV windows.
- In summary, this project tracks a book cover in real-time using color histogram back projection and the meanshift algorithm, providing visual feedback on tracking performance and history. It is useful for demonstrating object tracking techniques with OpenCV.

# To run this project, you need to have Python installed along with the following modules:
- numpy - for numerical operations
- opencv-python - for computer vision and video processing
- You can install the required modules using pip:
- pip install numpy opencv-python
  
After installing the dependencies, run the project by executing the Python script:


- python final_book_tracker.py
  
Make sure your webcam is connected and accessible, and the book cover image is present in the book_img directory as book_cover.jpg.

The program will prompt you to select the region of interest (ROI) on the book cover image for tracking. Then it will start the webcam feed and track the book cover in real-time, displaying tracking visualizations and performance metrics.

Press ESC to exit the tracking session.
