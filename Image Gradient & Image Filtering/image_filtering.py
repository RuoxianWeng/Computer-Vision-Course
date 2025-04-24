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

    # convert image to np array
    gray_image = np.array(gray_image, dtype=float)


    # Step 3: define the filter kernel as a numpy array
    kernel = np.array([[-1, 0, 1], 
                       [-2, 0, 2],
                       [-1, 0, 1]])


    # Step 4: filter the image with the kernel
    output = np.zeros((height, width))

    # gray image with zero padding 
    gray_image_with_padding = np.pad(gray_image, pad_width=1, constant_values=0)
    
    # fliter the image 
    for i in range(0, height):
        for j in range(0, width):
            sum = 0
            for k in range(3):
                for l in range(3):
                    sum += (gray_image_with_padding[i+k][j+l] * kernel[k][l])
            output[i][j] = sum


    # show result with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(gray_image, cmap = 'gray')
    ax.set_title('Original image')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap = 'gray')
    ax.set_title('Filtered image')

    plt.show()
