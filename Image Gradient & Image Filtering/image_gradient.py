import cv2
import numpy as np
import matplotlib.pyplot as plt

# main function
if __name__ == '__main__':
    # Step 1: read the crack box image with cv2.imread
    im = cv2.imread('cracker_box.jpg')
    height = im.shape[0]
    width = im.shape[1]


    # Step 2: use cv2.cvtColor to convert RGB image to gray scale image 
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    # Step 3: use central difference to compute image gradient on the gray scale image

    # convert image to np array
    gray_image = np.array(gray_image, dtype=float)
    
    # gray image with zero padding 
    gray_image_with_padding = np.pad(gray_image, pad_width=1, constant_values=0)

    gradient_x = np.zeros((height, width))
    gradient_y = np.zeros((height, width))
    
    # compute gradient using central difference 
    for x in range(1, height):
        for y in range(1, width):
            gradient_x[x-1][y-1] = (gray_image_with_padding[x][y+1] - gray_image_with_padding[x][y-1]) / 2
            gradient_y[x-1][y-1] = (gray_image_with_padding[x+1][y] - gray_image_with_padding[x-1][y]) / 2

    
    # show result with matplotlib 
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(gray_image, cmap = 'gray')
    ax.set_title('Original image')

    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(gradient_x, cmap = 'gray')
    ax.set_title('Gradient X')

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(gradient_y, cmap = 'gray')
    ax.set_title('Gradient Y')

    plt.show()
