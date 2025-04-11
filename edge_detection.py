import cv2
import numpy as np
from scipy import ndimage

def read_image(file):
    """Read uploaded image (BytesIO) into OpenCV format"""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_sobel(image):
    gray = to_grayscale(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    sobelxy = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=5)
    return cv2.convertScaleAbs(sobelxy)

def apply_prewitt(image):
    gray = to_grayscale(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_x = cv2.filter2D(blur, -1, kernel_x)
    prewitt_y = cv2.filter2D(blur, -1, kernel_y)
    prewitt = prewitt_x + prewitt_y
    return cv2.convertScaleAbs(prewitt)

def apply_roberts(image):
    gray = to_grayscale(image)
    gx = np.array([[1, 0], [0, -1]])
    gy = np.array([[0, 1], [-1, 0]])
    grad_x = ndimage.convolve(gray, gx)
    grad_y = ndimage.convolve(gray, gy)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edges = (magnitude > 10) * 255
    return edges.astype(np.uint8)

def apply_compass(image):
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    compass_kernels = [
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
        # Add more compass kernels if needed
    ]

    responses = [cv2.filter2D(gradient_x, -1, k) + cv2.filter2D(gradient_y, -1, k) for k in compass_kernels]
    edge_map = np.max(responses, axis=0)
    edges = (edge_map > 50) * 255
    return edges.astype(np.uint8)

def apply_log(image):
    gray = to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    marr_hildreth = cv2.convertScaleAbs(abs_laplacian)
    return marr_hildreth
