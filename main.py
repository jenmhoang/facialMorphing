import imageio
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline

def get_points(im, num_points):
    plt.imshow(im)
    points = plt.ginput(num_points, -1)
    plt.close()
    return points

def get_affine_transform(tri1, tri2):
    tri1_M = np.vstack([np.transpose(tri1), np.ones(len(tri1))])
    tri2_M = np.vstack([np.transpose(tri2), np.ones(len(tri2))])
    transform = np.matmul(tri2_M, np.linalg.inv(tri1_M))[:2]

    dim3 = np.zeros(transform.shape[1])
    dim3[dim3.shape[0] - 1] = 1
    return np.vstack((transform, dim3))

def apply_transform(transform, input_tri, f, w, im_out):
    r, c = draw.polygon(input_tri.T[0], input_tri.T[1])
    matrix = np.vstack([r, c, np.ones(r.shape[0])])
    tri_warped = np.matmul(transform, matrix).astype(int)

    r = np.clip(r, 0, im_out.shape[2] - 1)
    c = np.clip(c, 0, im_out.shape[1] - 1)

    for i in range(3):
        im_out[i][c, r] += w * f[i](tri_warped[1], tri_warped[0], grid=False)
    
def morph(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    # output image
    im_warped = np.zeros((3, im2.shape[0], im2.shape[1]))

    ## compute weighted mean points and triangulation
    mean_points = np.average(np.array([im1_pts, im2_pts]), axis=0, weights=np.array([1 - warp_frac, warp_frac]))
    mean_triangulation = Delaunay(mean_points)

    ## color matching function
    im_f = np.empty(3, dtype=RectBivariateSpline)
    target_f = np.empty(3, dtype=RectBivariateSpline)
    for i in range(3):
        im_f[i] = RectBivariateSpline(np.arange(im1.shape[0]), np.arange(im1.shape[1]), im1[:,:,i])
        target_f[i] = RectBivariateSpline(np.arange(im2.shape[0]), np.arange(im2.shape[1]), im2[:,:,i])

    ## warp image via affine transform
    for indices in mean_triangulation.simplices:
        im_tri = np.array([im1_pts[i] for i in indices])
        target_tri = np.array([im2_pts[i] for i in indices])
        mean_tri = np.array([mean_points[i] for i in indices])

        im_transform = get_affine_transform(mean_tri, im_tri)
        target_transform = get_affine_transform(mean_tri, target_tri)

        apply_transform(im_transform, mean_tri, im_f, 1 - dissolve_frac, im_warped)
        apply_transform(target_transform, mean_tri, target_f, dissolve_frac, im_warped)

    ## combine color channels to create output image
    warped_image = np.dstack(im_warped)
    return warped_image

def main():
    ## input points files
    IM1_POINTS_PATH = "im_points.txt"
    IM2_POINTS_PATH = "target_points.txt"

    ## input images
    im = plt.imread("images/me.png")
    target = plt.imread("images/jess.png")

    ## get points via input or text file
    if (not (Path(IM1_POINTS_PATH).is_file() and Path(IM2_POINTS_PATH).is_file())):
        np.savetxt(IM1_POINTS_PATH, get_points(im, 52))
        np.savetxt(IM2_POINTS_PATH, get_points(target, 52))
    points = np.array([np.genfromtxt(IM1_POINTS_PATH), np.genfromtxt(IM2_POINTS_PATH)])

    step = 1/46 # get 46 frames
    frames = []
    ## morph with weight w from 0 to 1 over 0.05 size steps
    for w in np.arange(0, 1.1, step):
        warped_image = morph(im, target, points[0], points[1], w, w)
        frames.append(warped_image)
    
    imageio.mimsave("morphing.gif", frames, format='GIF', fps=30)

if __name__ == "__main__":
    main()
