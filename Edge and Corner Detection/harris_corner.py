"""
CS 4391 Homework 2 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#TODO: implement this function
# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image 
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):

    mask = np.zeros(R.shape)
    R_with_padding = np.pad(R, pad_width=1, constant_values=0)

    for i in range(1, R_with_padding.shape[0]-2): 
        for j in range(1, R_with_padding.shape[1]-2): 
            # if already 0, ignore 
            if R_with_padding[i][j] == 0: 
                continue

            # check if pixel is a local max: 
            if ((R_with_padding[i][j] >= R_with_padding[i][j-1]) and (R_with_padding[i][j] >= R_with_padding[i][j+1]) and 
                (R_with_padding[i][j] >= R_with_padding[i-1][j]) and (R_with_padding[i][j] >= R_with_padding[i+1][j]) and 
                (R_with_padding[i][j] >= R_with_padding[i+1][j-1]) and (R_with_padding[i][j] >= R_with_padding[i-1][j+1]) and 
                (R_with_padding[i][j] >= R_with_padding[i+1][j+1]) and (R_with_padding[i][j] >= R_with_padding[i-1][j-1])): 
                    mask[i][j] = 1

    return mask


#TODO: implement this function
# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# You can use opencv functions and numpy functions
def harris_corner(im):

    # step 0: convert RGB to gray-scale image
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.0
    
    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    # step 2: compute products of derivatives at every pixels
    gradient_x_squared = gradient_x * gradient_x
    gradient_y_squared = gradient_y * gradient_y
    gradient_product = gradient_x * gradient_y

    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter from OpenCV
    sum_of_prod_xx = cv2.GaussianBlur(gradient_x_squared, (5, 5), 0)
    sum_of_prod_yy = cv2.GaussianBlur(gradient_y_squared, (5, 5), 0)
    sum_of_prod_xy = cv2.GaussianBlur(gradient_product, (5, 5), 0)

    # step 4: compute determinant and trace of the M matrix
    determinant = (sum_of_prod_xx * sum_of_prod_yy) - (sum_of_prod_xy * sum_of_prod_xy)
    trace = sum_of_prod_xx + sum_of_prod_yy

    # step 5: compute R scores with k = 0.05
    k = 0.05
    R = determinant - k * (trace * trace)

    # step 6: thresholding
    # up to now, you shall get a R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0
    
    # step 7: non-maximum suppression
    #TODO implement the non_maximum_suppression function above
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'cracker_box.jpg'
    im = cv2.imread(rgb_filename)
    
    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)
    
    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()
        
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5, linewidth=0)
    ax.set_title('our corner image')
    
    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5, linewidth=0)
    ax.set_title('opencv corner image')

    plt.show()