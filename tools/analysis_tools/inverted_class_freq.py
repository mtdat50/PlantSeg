from glob import glob
import numpy as np
from PIL import Image

if __name__ == "__main__":
    wheat_mask_dir = "../../datasets/Plantseg/annotations/train/*"
    wheat_mask_paths = glob(wheat_mask_dir)
    print(f"Found {len(wheat_mask_paths)} wheat mask images.")

    classes_cnt = [0 for _ in range(9)]
    total = 0
    for path in wheat_mask_paths:
        image = np.array(Image.open(path))
        total += np.prod(image.shape)

        for i in range(len(classes_cnt)):
            classes_cnt[i] += np.sum(image == i)

    classes_inverted_freq = [total / cnt for cnt in classes_cnt]
    np.set_printoptions(precision=2, suppress=True)
    sqrt_classes_inverted_freq = np.sqrt(classes_inverted_freq)
    log_classes_inverted_freq = np.log(classes_inverted_freq)
    print(classes_inverted_freq)
    print(sqrt_classes_inverted_freq)
    print(log_classes_inverted_freq)
