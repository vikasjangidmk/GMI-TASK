import cv2
import numpy as np
from scipy.ndimage import rotate  # Updated import to avoid deprecation warning

def correct_skew_hough(image, delta=0.5, limit=15):
    """Enhanced skew correction with noise reduction and adaptive thresholding"""
    def determine_score(arr, angle):
        # Updated rotate function from scipy.ndimage
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        return np.sum((histogram[1:] - histogram[:-1]) ** 2)

    # Handle color/grayscale input
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Pre-processing for better thresholding
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Find optimal rotation angle
    angles = np.arange(-limit, limit + delta, delta)
    scores = [determine_score(thresh, angle) for angle in angles]
    best_angle = angles[np.argmax(scores)]

    # Rotate and return image
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), best_angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    return best_angle, corrected

def preprocess_document(image):
    """Complete document preprocessing pipeline"""
    # Handle different formats
    if len(image.shape) == 3 and image.shape[2] == 4:  # PNG with alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 1. Initial noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 2. Skew correction
    angle, deskewed = correct_skew_hough(denoised)
    
    # 3. Normalization and contrast enhancement
    lab = cv2.cvtColor(deskewed, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    normalized_l = clahe.apply(l_channel)
    normalized_lab = cv2.merge([normalized_l, a, b])
    result = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    
    return angle, result

# Main execution with GUI display removed
if __name__ == '__main__':
    image_path = r'C:\Users\vikas\OneDrive\Desktop\GMI-TASK\skew_check\doc_skew.png'
    output_path = r'C:\Users\vikas\OneDrive\Desktop\GMI-TASK\skew_check\corrected_final_best.png'

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Image not found at {image_path}")
    else:
        # Using the full preprocessing pipeline
        angle, corrected = preprocess_document(image)
        
        print(f"Final skew angle: {angle:.2f}°")
        # Save the image without displaying
        cv2.imwrite(output_path, corrected)
        print(f"Corrected image saved to: {output_path}")