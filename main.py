import sys
import numpy as np
from skimage import io
import skimage.feature as sf
import matplotlib.pyplot as plt
from sympy import *
import cv2 as cv
from scipy import ndimage

# 1
picture = "picture.png"
cam = cv.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    if cv.waitKey(1) == 27:
        cv.imwrite(picture, img)
        break
    elif img is None:
        sys.exit('Couldn`t read the image')
        break
    cv.imshow('My Webcam', img)

del cam

# 2
face_cv = cv.imread(picture)
cv.imshow('Image', face_cv)
cv.waitKey(0)

# 3-a
fig, ax = plt.subplots()
face = io.imread(picture)
plt.imshow(face)
plt.show()
points, = ax.plot(40, 200, 'r*')
x, y = points.get_data()
print(x, y)

# 3-b
image = face
image = image[x[0]:x[0]+10, y[0]:y[0]+10]
plt.imshow(image)
plt.show()
cv.waitKey(0)

# 3-c
ired = np.zeros(face.shape)
ired[:, :, 0] = face[:, :, 0]

igreen = np.zeros(face.shape)
igreen[:, :, 1] = face[:, :, 1]

iblue = np.zeros(face.shape)
iblue[:, :, 2] = face[:, :, 2]

cv.imshow('Blue', ired/255)
cv.imshow('Green', igreen/255)
cv.imshow('Red', iblue/255)
cv.waitKey(0)

# 4
gray = cv.cvtColor(face_cv, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)
cv.waitKey(0)

# 5
kernel = np.ones((3, 3), np.float32)/25
print(kernel)
filtered_image = cv.filter2D(gray, -1, kernel)
cv.imshow('Original Image', gray)
cv.imshow('Blurred Image', filtered_image)
gaussisan = cv.GaussianBlur(gray, (3, 3), 0)
cv.imshow('Gaussian Blur', gaussisan)
cv.waitKey(0)

#6
edges = cv.Laplacian(filtered_image, -1, ksize=5, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
cv.imshow('Edges', edges)
cv.waitKey(0)

# 7
#Magnitude
f = np.fft.fft2(edges)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.imshow(magnitude_spectrum, cmap = 'gray')
#Phase
dft = np.fft.fft2(edges)
dft_shift = np.fft.fftshift(dft)
phase_spectrum = np.angle(dft_shift)
cv.imshow("Phase", phase_spectrum)
plt.show()
cv.waitKey(0)

# 8
plt.hist(face.ravel(), 256, [0, 256])
plt.show()

# 9
mask = np.zeros(face.shape[:2], np.uint8)
mask[200:450, 150:450] = 255
masked_img = cv.bitwise_and(face_cv,face_cv,mask = mask)
cv.imshow('Masked Image', masked_img)
cv.waitKey(0)

# 10
rotated = ndimage.rotate(face_cv, 60)
cv.imshow("Rotated 60 degrees", rotated)

h_flip = cv.flip(face_cv, 1)
cv.imshow("Horizontal Flip", h_flip)

v_flip = cv.flip(face_cv, 0)
cv.imshow("Vertical FLip", v_flip)

cv.waitKey(0)
cv.destroyAllWindows()