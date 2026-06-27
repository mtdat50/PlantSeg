import cv2
import numpy as np



if __name__ == "__main__":
    image_name = "wheat_stripe_rust_Baidu_0250"
    image_path = f"data/plantsegwheat/images/train/{image_name}.jpg"
    mask_path = f"data/plantsegwheat/annotations/train/{image_name}.png"

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]
    for row in range(h):
        for col in range(w):
            if mask[row, col]:
                image[row, col] = np.add(image[row, col], (500, 0, 300)) / 3

    cv2.imwrite(f"{image_name}_overlay.jpg", image)
