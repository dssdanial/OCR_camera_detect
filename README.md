# OCR_camera_detect

This project demonstrates the use of OpenCV and image processing techniques to prepare images for Optical Character Recognition (OCR). The steps include pre-processing methods such as grayscale conversion, thresholding, erosion, and dilation to enhance the clarity of text in images for better OCR results.

### Steps in Pre-Processing

#### 1. Displaying Images Without Distortion

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline  

def Display(input_path):
    dpi = 79
    im_data = plt.imread(input_path)
    height, width = im_data.shape[:2]
    re_size = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=re_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()

image_path = "data_textbook/page_01.jpg"
img = cv2.imread(image_path)
Display(image_path)
```

#### 2. Image Inversion

```python
image_invert = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted.jpg", image_invert)
Display("temp/inverted.jpg")
```

#### 3. Binary Image Conversion

1. Convert to Grayscale:

```python
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('temp/gray_scale.jpg', grayscale_img)
```

2. Convert to Black and White:

```python
threshold, bin_img = cv2.threshold(grayscale_img, 210, 230, cv2.THRESH_BINARY)
cv2.imwrite('temp/binary_img.jpg', bin_img)
Display("temp/binary_img.jpg")
```

#### 4. Erosion and Dilation

Erosion and dilation techniques are applied to improve OCR performance by enhancing text visibility and reducing noise.

- **Erosion** removes small artifacts and separates connected characters.
- **Dilation** thickens weak or faint text for better OCR recognition.

##### Erosion Followed by Dilation:

```python
kernel1 = np.ones((1, 1), np.uint8)
image1 = cv2.erode(bin_img, kernel1, iterations=1)
image1 = cv2.dilate(image1, kernel1, iterations=1)
image1 = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, kernel1)
image1 = cv2.medianBlur(image1, 3)

cv2.imwrite('temp/Er_Di.jpg', image1)
Display("temp/Er_Di.jpg")
```

##### Dilation Followed by Erosion:

```python
kernel2 = np.ones((1, 1), np.uint8)
image2 = cv2.dilate(bin_img, kernel2, iterations=1)
image2 = cv2.erode(image2, kernel2, iterations=1)
image2 = cv2.morphologyEx(image2, cv2.MORPH_CLOSE, kernel2)
image2 = cv2.medianBlur(image2, 3)

cv2.imwrite('temp/Di_Er.jpg', image2)
Display("temp/Di_Er.jpg")
```
