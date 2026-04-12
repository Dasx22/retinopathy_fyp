import cv2
import glob
import numpy as np
import os

def scaleRadius(img, scale):
    # Note: Using '//' for Python 3 integer division compatibility
    x = img[img.shape[0]//2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)

def preprocess_images(input_dir, output_dir, scale=300):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all JPEGs from the input directory
    image_paths = glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    for f in image_paths:
        try:
            a = cv2.imread(f)
            if a is None:
                continue
                
            # 1. Scale image to a given radius
            a = scaleRadius(a, scale)
            
            # 2. Subtract local mean color (maps local average to 50% gray)
            a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), int(scale/30)), -4, 128)
            
            # 3. Remove outer 10% to eliminate boundary effects
            b = np.zeros(a.shape)
            cv2.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
            a = a * b + 128 * (1 - b)
            
            # Save to output directory
            basename = os.path.basename(f)
            cv2.imwrite(os.path.join(output_dir, basename), a)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

# Run the preprocessing on your sample folder
# preprocess_images(input_dir="sample", output_dir="sample_preprocessed", scale=300)

# for train folder
# preprocess_images(input_dir="train", output_dir="train_preprocessed", scale=300)

# for test folder
preprocess_images(input_dir="test", output_dir="test_preprocessed", scale=300)