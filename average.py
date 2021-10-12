import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from main import get_affine_transform, apply_transform, get_points, morph
from scipy.spatial import Delaunay

## text point locations for imm_face_db
START_POINTS = 16
END_POINTS = 74

POINT1_LINESTART = 6
POINT2_LINESTART = 18
POINT_CHAR_LENGTH = 10

DATASET_POINTS_PER_IMAGE = 58

def parse_image_points(directory):
    files = os.listdir(directory)
    all_points = {}
    for i in range(len(files)):
        if (files[i][0] != '.' and files[i][-5:-4] == "m"): # is a male
            points = np.empty((DATASET_POINTS_PER_IMAGE + 4, 2)) # additional 4 spaces allocated for corners (to be added later once we know image size)
            lines = []
            
            with open(directory + "/" + files[i]) as f:
                lines = f.readlines()
            
            for p in range(START_POINTS, END_POINTS):
                points[p - START_POINTS] = np.array([float(lines[p][POINT1_LINESTART : POINT1_LINESTART+POINT_CHAR_LENGTH]),
                                                    float(lines[p][POINT2_LINESTART : POINT2_LINESTART+POINT_CHAR_LENGTH])])

            ## add corner points
            corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            points[-4:] = corners
            all_points[files[i][:-4]] = points # file name without .asf is at substring :-4

    return all_points
            
def warp_to_mean(images, points, mean_triangulation, mean_points):
    all_warped_images = []

    for k in images:
        im = images[k]
        pts = points[k]
        pts *= np.array([im.shape[1], im.shape[0]])
        
        ## output image
        im_warped = np.zeros((3, im.shape[0], im.shape[1]))

        ## color matching function
        f = np.empty(3, dtype=RectBivariateSpline)
        for i in range(3):
            f[i] = RectBivariateSpline(np.arange(im.shape[0]), np.arange(im.shape[1]), im[:,:,i])
        
        ## warp to image via affine transform
        for indices in mean_triangulation.simplices:
            im_tri = np.array([pts[i] for i in indices])
            mean_tri = np.array([mean_points[i] for i in indices])

            im_transform = get_affine_transform(mean_tri, im_tri)
            apply_transform(im_transform, mean_tri, f, 1, im_warped)
        
        warped_image = np.dstack(im_warped) / 255
        plt.imsave("results/warped_to_mean/"+k+".png", np.clip(warped_image, 0, 1))
        all_warped_images.append(warped_image)

    return all_warped_images

def main():
    ## database directories
    images_dir = "imm_face_db/images"
    points_dir = "imm_face_db/datapoints"

    ## collect images and points
    images = {}
    for im in os.listdir(images_dir):
        if (im[0] != '.'):
            file = images_dir + "/" + im
            if file[19:24][-1:] == "m": # if the image is of a male
                images[file[19:24]] = plt.imread(file) # file name without .jpg and directory names is at substring 19:24

    points = parse_image_points(points_dir)

    ## compute mean points and triangulation
    mean_points = np.mean(np.array(list(points.values())), axis=0)
    im_shape = images['01-1m'].shape # one of the image files in the dictionary
    mean_points *= np.array([im_shape[1], im_shape[0]])
    mean_triangulation = Delaunay(mean_points)

    ## warp all images to mean
    warped_images = warp_to_mean(images, points, mean_triangulation, mean_points)

    ## combine warped faces to create average face
    mean_image = np.zeros(im_shape)
    weight = 1 / len(warped_images)
    for im in warped_images:
        mean_image += im * weight

    ## read in input image
    im = plt.imread("images/me.png")
    im_points = get_points(im, 62)

    ## warp im to mean image and vice versa
    im_to_mean = morph(im, mean_image, im_points, mean_points, 1, 0)
    plt.imsave("results/im_to_mean.png", np.clip(im_to_mean, 0, 1))

    mean_to_im = morph(mean_image, im, mean_points, im_points, 1, 0)
    plt.imsave("results/mean_to_im.png", np.clip(mean_to_im, 0, 1))

    im_to_mean = morph(im, mean_image, im_points, mean_points, 1.5, 0)
    plt.imsave("results/extrapolated.png", np.clip(im_to_mean, 0, 1))

if __name__ == "__main__":
    main()