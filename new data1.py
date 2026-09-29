# mm 

import cv2
import numpy as np
import json


def load_image(image_path):
    """Load an image from disk."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return image


def detect_wound_region(image):
    """
    Detect the wound region using color-based segmentation.
    Returns the mask and contour of the largest detected region.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad skin-wound color range (red/pink tones) - tuned loosely for now
    lower_bound = np.array([0, 30, 30])
    upper_bound = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower_bound, upper_bound)

    lower_bound2 = np.array([160, 30, 30])
    upper_bound2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)

    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, None

    largest_contour = max(contours, key=cv2.contourArea)
    return mask, largest_contour


def measure_wound_area(contour):
    """Return wound area in pixels."""
    if contour is None:
        return 0
    return cv2.contourArea(contour)


def classify_tissue_type(image, mask):
    """
    Classify dominant tissue type based on color inside the wound mask.
    Returns: 'granulation' (red), 'slough' (yellow), or 'necrotic' (black)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    wound_pixels = hsv[mask > 0]

    if len(wound_pixels) == 0:
        return "unknown"

    avg_hue = np.mean(wound_pixels[:, 0])
    avg_value = np.mean(wound_pixels[:, 2])

    if avg_value < 50:
        return "necrotic"
    elif 20 <= avg_hue <= 40:
        return "slough"
    else:
        return "granulation"


def draw_segmentation(image, contour, output_path):
    """Draw the detected wound contour on the image and save it."""
    output = image.copy()
    if contour is not None:
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
    cv2.imwrite(output_path, output)
    return output_path


def analyze_wound(image_path, output_path="data/images/segmented_output.jpg"):
    """
    Full pipeline: load image, detect wound, measure area, classify tissue.
    Returns a dict of results.
    """
    image = load_image(image_path)
    mask, contour = detect_wound_region(image)
    area = measure_wound_area(contour)
    tissue_type = classify_tissue_type(image, mask)
    draw_segmentation(image, contour, output_path)

    results = {
        "image_path": image_path,
        "wound_area_pixels": float(area),
        "tissue_type": tissue_type,
        "segmented_image_path": output_path,
    }
    return results


def compare_wounds(image_path_1, image_path_2):
    """Compare two wound images and calculate healing delta."""
    result1 = analyze_wound(image_path_1, "data/images/segmented_1.jpg")
    result2 = analyze_wound(image_path_2, "data/images/segmented_2.jpg")

    area1 = result1["wound_area_pixels"]
    area2 = result2["wound_area_pixels"]

    if area1 == 0:
        percent_change = 0
    else:
        percent_change = ((area1 - area2) / area1) * 100

    comparison = {
        "before": result1,
        "after": result2,
        "area_change_percent": round(percent_change, 2),
        "status": "improving" if percent_change > 0 else "worsening" if percent_change < 0 else "stable",
    }
    return comparison


if __name__ == "__main__":
    print("Starting wound analysis...")
    test_image_path = r"C:\Users\manoj\Downloads\wound.jpg"
    print(f"Loading image from: {test_image_path}")
    results = analyze_wound(test_image_path)
    print("Analysis complete!")
    print(json.dumps(results, indent=2))