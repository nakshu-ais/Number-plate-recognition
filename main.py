# PACKAGES NEEDED
import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils

# IMAGE PROCESSING

# reading the image
img = cv2.imread("testMain.jpg")
if img is None:
    raise FileNotFoundError("Could not read testMain.jpg")

# converting image to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# noise reductions
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# EDGE DETECTION
# Using the canny edge detection algorithm -
# smoothening the photo and scanning for areas where brightness changes sharply
edged = cv2.Canny(bfilter, 30, 200)

# FINDING CONTOURS
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
# loop over contours to find best possible contour of 10 contours
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
# output is the coordinates of the number plate
print("location: ", location)
if location is None:
    raise RuntimeError("Could not detect a license plate in testMain.jpg")

# MASKING OUT NUMBER PLATE AREA
# blank image with same dimensions as original image
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask,[location],0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# CROPPING OF THE NUMBER PLATE
# 4-corner coordinates
(x,y) = np.where(mask == 255)
# top left corner
(x1,y1) = (np.min(x), np.min(y))
# bottom right corner
(x2,y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
if cropped_image.size == 0:
    raise RuntimeError("The detected license plate crop is empty")

# EXTRACT TEXT FROM IMAGES WITH OCR
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

# DISPLAY FINAL OUTPUT
if result:
    text = result[0][-2]
else:
    text = "No text detected"
print(f"The number plate is {text}")

res = img.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(
    res,
    text=text,
    org=(int(approx[0][0][0]), int(approx[1][0][1]) + 60),
    fontFace=font,
    fontScale=1,
    color=(0, 255, 0),
    thickness=2,
    lineType=cv2.LINE_AA,
)
# creating a rectangle around the text
res = cv2.rectangle(res, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

# DISPLAY THE IMAGE STAGES, CROPPED PLATE PROFILE, AND HISTOGRAM
# The center row gives a one-dimensional gray-level profile across the crop.
center_row = cropped_image[cropped_image.shape[0] // 2, :]
figure, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Grayscale Image")
axes[0, 2].imshow(bfilter, cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title("Processed Image")

axes[1, 0].imshow(edged, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("Edge Detection")
axes[1, 1].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Detected Plate Region")
axes[1, 2].imshow(cropped_image, cmap="gray", vmin=0, vmax=255)
axes[1, 2].set_title("Cropped License Plate (Grayscale)")

axes[2, 0].plot(np.arange(center_row.size), center_row, color="navy")
axes[2, 0].set_title("Cropped Plate Gray-Level Profile (Center Row)")
axes[2, 0].set_xlabel("Horizontal pixel position")
axes[2, 0].set_ylabel("Gray level (0–255)")
axes[2, 0].set_ylim(0, 255)
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].hist(cropped_image.ravel(), bins=256, range=(0, 256), color="dimgray")
axes[2, 1].set_title("Cropped Plate Grayscale Histogram")
axes[2, 1].set_xlabel("Gray level (0–255)")
axes[2, 1].set_ylabel("Pixel count")
axes[2, 1].set_xlim(0, 255)

axes[2, 2].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title(f"OCR Result: {text}")

for row in axes:
    for axis in row:
        axis.title.set_fontsize(10)
        axis.set_xticks([]) if axis.images else None
        axis.set_yticks([]) if axis.images else None

plt.show()

