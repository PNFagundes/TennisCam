import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0) # Capture image from webcam

_, frame = cap.read()
drawn_frame = np.copy(frame)
tracker = np.copy(frame) # Track the pen's movement

last_seen = time.time() # Track the time

while True:
    _, frame = cap.read() # Read the image
        
    frame_original = np.copy(frame) # Copy the original image

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV
    
    # Define color ranges
    lower = np.array([100, 50, 50]) 
    upper = np.array([140, 255, 255])
    
    # Enhance the drawing
    mask = cv2.inRange(hsv, lower, upper) # Create mask
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    filter_imagem = cv2.bitwise_and(frame , frame , mask=mask) # Filter the image
    
    cv2.imshow("Original Image", frame_original)
    
    cv2.imshow("Image Filtered", filter_imagem)
    
    # Counting non-zero pixels
    aux = np.copy(filter_imagem)
    aux_gray = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)
    nonzeroPixels = np.count_nonzero(aux_gray)
    print(f"Counting non-zero pixels - {nonzeroPixels}")
    
    # Establishing some conditions for the drawing
    if nonzeroPixels != 0:
        drawn_frame = np.copy(frame_original) # Copying the last image
        mask_tracker = np.all(tracker == [255, 255, 255], axis=-1) # Creating a mask of the drawing
        # Updating the drawn image
        drawn_frame[mask_tracker] = [255, 255, 255] 
        drawn_frame[aux_gray!=0] = [255,255,255] 
        tracker[aux_gray!=0] = [255,255,255] # Keeping a record of the drawing
        last_seen = time.time() # Keeping a record of the time

    else:
        if time.time() - last_seen > 3: # Checking the time
            # Delete the drawing when the time expires
            drawn_frame = np.copy(frame_original)
            tracker = np.copy(frame_original)
        else:
            drawn_frame = np.copy(frame_original)
            mask_tracker = np.all(tracker == [255, 255, 255], axis=-1)
            drawn_frame[mask_tracker] = [255, 255, 255]
            tracker[aux_gray!=0] = [255,255,255]
        
    drawn_frame = cv2.flip(drawn_frame, 1)        
    cv2.imshow("Drawn Image", drawn_frame)
    
    # Exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

    # Save the image
    if key == ord('c'):
        cv2.imwrite('img_saved.png', frame_original)
    
cap.release() # Release the webcam resource
cv2.destroyAllWindows() # Close the windows
