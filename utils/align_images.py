import cv2
import numpy as np

def align_images(modern_img_path, historical_img_path, output_path="aligned_scene.png"):
    modern_img = cv2.imread(modern_img_path)
    historical_img = cv2.imread(historical_img_path)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(modern_img, None)
    kp2, des2 = sift.detectAndCompute(historical_img, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        aligned_img = cv2.warpPerspective(historical_img, matrix, (modern_img.shape[1], modern_img.shape[0]))
        cv2.imwrite(output_path, aligned_img)
        print(f"Aligned image saved at {output_path}")
    else:
        print("Not enough matches found to align images.")

# Example usage
if __name__ == "__main__":
    align_images("modern_scene.jpg", "historical_scene.png")
