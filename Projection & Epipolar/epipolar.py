import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from backproject import backproject
    
    
# read rgb, depth, mask and meta data from files
def read_data(file_index):

    # read the image in data
    # rgb image
    rgb_filename = 'data/%06d-color.jpg' % file_index
    im = cv2.imread(rgb_filename)
    
    # depth image
    depth_filename = 'data/%06d-depth.png' % file_index
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    depth = depth / 1000.0
    
    # read the mask image
    mask_filename = 'data/%06d-label-binary.png' % file_index
    mask = cv2.imread(mask_filename)
    mask = mask[:, :, 0]
    
    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    # load matedata
    meta_filename = 'data/%06d-meta.mat' % file_index
    meta = scipy.io.loadmat(meta_filename)
    
    return im, depth, mask, meta
    
    
# compute the fundamental matrix
# xy1 and xy2 are with shape (n, 2)
def compute_fundamental_matrix(xy1, xy2):

    # step 1: construct the A matrix

    # get homogenous coordinates 
    homo_xy1 = np.hstack((xy1, np.ones((xy1.shape[0], 1))))
    homo_xy2 = np.hstack((xy2, np.ones((xy2.shape[0], 1))))

    # construct A matrix
    A = np.zeros((xy1.shape[0], 9))
    for i in range(xy1.shape[0]): # for each correspondence
        x = homo_xy1[i]
        x_prime = homo_xy2[i]
        A_i = []
        for u in range(homo_xy2.shape[1]): 
            for v in range(homo_xy1.shape[1]): 
                A_i.append(x_prime[u] * x[v])
        A[i] = A_i

    # step 2: SVD of A
    # use numpy function for SVD
    U, D, V_T = np.linalg.svd(A)

    # step 3: get the last column of V
    # last row of V_T
    F = V_T[-1]
    F = F.reshape(3, 3)
    
    # step 4: SVD of F
    U, D, V_T = np.linalg.svd(F)
    # step 5: mask the last element of singular value of F
    D[-1] = 0
    
    # step 6: reconstruct F
    F = np.dot(U, np.dot(np.diag(D), V_T))

    return F  


# main function
if __name__ == '__main__':

    # read image 1
    im1, depth1, mask1, meta1 = read_data(6)
    
    # read image 2
    im2, depth2, mask2, meta2 = read_data(7)
    
    # intrinsic matrix
    intrinsic_matrix = meta1['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
        
    # get the point cloud from image 1
    pcloud = backproject(depth1, intrinsic_matrix)
    
    # find the boundary of the mask 1
    boundary = np.where(mask1 > 0)
    x1 = np.min(boundary[1])
    x2 = np.max(boundary[1])
    y1 = np.min(boundary[0])
    y2 = np.max(boundary[0])
    
    # sample n pixels (x, y) inside the bounding box of the cracker box 
    n = 10
    height = im1.shape[0]
    width = im1.shape[1]
    x = np.random.randint(x1, x2, n)
    y = np.random.randint(y1, y2, n)
    index = np.zeros((n, 2), dtype=np.int32)
    index[:, 0] = x
    index[:, 1] = y
    print(index, index.shape)

    # get the coordinates of the n pixels
    pc1 = np.ones((4, n), dtype=np.float32)
    for i in range(n):
        x = index[i, 0]
        y = index[i, 1]
        print(x, y)
        pc1[:3, i] = pcloud[y, x, :]
    print('pc1', pc1)
    
    # filter zero depth pixels
    ind = pc1[2, :] > 0
    pc1 = pc1[:, ind]
    index = index[ind]
    xy1 = index
    # xy1 is a set of pixels on image 1
    # we will find the correspondences of these pixels
    
    # transform the points to another camera
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    print(RT1.shape, RT2.shape)
    
    # find the correspondences of xy1
    # let the corresponding pixels on image 2 be xy2 with shape (n, 2)

    # get 3d camera 1 coordinates for the set of pixels in image 1
    index_3d_im1 = np.zeros((xy1.shape[0], 3))
    for i in range(xy1.shape[0]): 
        pixel = xy1[i]
        index_3d_im1[i] = pcloud[pixel[1]][pixel[0]]
    
    # transform camera 1 view to world view 
    RT1_inv = np.linalg.inv(RT1)
    for i in range(len(index_3d_im1)): 
        homo_coordinate = np.append(index_3d_im1[i], 1)
        homo_coordinate = np.dot(RT1_inv, homo_coordinate)
        index_3d_im1[i] = homo_coordinate[:-1]

    # transform world view to camera 2 view
    index_3d_im2 = np.zeros(index_3d_im1.shape)
    for i in range(len(index_3d_im1)): 
        homo_coordinate1 = np.append(index_3d_im1[i], 1)
        homo_coordinate2 = np.dot(RT2, homo_coordinate1)
        index_3d_im2[i] = homo_coordinate2[:-1]
    
    # project the transformed 3D points to the second image
    xy2 = np.zeros((xy1.shape[0], 2))
    for i in range(len(index_3d_im2)): 
        homo_coordinate = np.dot(intrinsic_matrix, index_3d_im2[i])
        xy2[i] = [homo_coordinate[0]/homo_coordinate[2], homo_coordinate[1]/homo_coordinate[2]]

    F = compute_fundamental_matrix(xy1, xy2)
    
    # visualization for debugging
    fig = plt.figure()
        
    # show RGB image 1 and sampled pixels
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: correspondences', fontsize=15)
    plt.scatter(x=xy1[:, 0], y=xy1[:, 1], c='y', s=20)
    
    # show RGB image 2 and sampled pixels
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: correspondences', fontsize=15)
    plt.scatter(x=xy2[:, 0], y=xy2[:, 1], c='g', s=20)
    
    # show three pixels on image 1
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: sampled pixels', fontsize=15)
    
    # compute epipolar lines of three sampled points
    px = 233
    py = 145
    p = np.array([px, py, 1]).reshape((3, 1))
    l1 = np.matmul(F, p)
    print(p.shape)
    print(l1) 
    plt.scatter(x=px, y=py, c='r', s=40)
    
    px = 240
    py = 245
    p = np.array([px, py, 1]).reshape((3, 1))
    l2 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='g', s=40)
    
    px = 326
    py = 268
    p = np.array([px, py, 1]).reshape((3, 1))
    l3 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='b', s=40)    
    
    # draw the epipolar lines of the three pixels
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: epipolar lines', fontsize=15)
    
    for x in range(width):
        y1 = (-l1[0] * x - l1[2]) / l1[1]
        if y1 > 0 and y1 < height-1:
            plt.scatter(x, y1, c='r', s=1)
            
        y2 = (-l2[0] * x - l2[2]) / l2[1]
        if y2 > 0 and y2 < height-1:
            plt.scatter(x, y2, c='g', s=1)
            
        y3 = (-l3[0] * x - l3[2]) / l3[1]
        if y3 > 0 and y3 < height-1:
            plt.scatter(x, y3, c='b', s=1)                        
                  
    plt.show()
