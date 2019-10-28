"""Script for compressing images using the K-means algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from PIL import Image
from tqdm import tqdm


class KMeans:
    
    def __init__(self, K):
        """Initializes a k-means object that supports training and clustering

        Args:
            K: (int) number of clusters or colors in a compressed image
        """
        self.K = K

    def _load_image(self, img_filename):
        """Loads in an image and reshapes it for processing

        Args:
            img_filename: (str) path to image file
        """
        img = Image.open(img_filename)
        img = np.asarray(img) / 255.

        w, h, c = img.shape
        return img.reshape((w * h, c)), img.shape

    def _image_from_numpy(self, img, im_size):
        """Converts a numpy array into a PIL image

        Args:
            img: (np.ndarray) image as numpy float array
            im_size: (tuple) image size (WHC) format
        """
        # reconstruct the image by assigning the quantized pixels to the labels
        image_reconstructed = (img * 255).astype(np.uint8)
        image_reconstructed = image_reconstructed.reshape(im_size)

        return Image.fromarray(image_reconstructed)

    def _init_means(self, img):
        """Randomly initializes K means

        Args:
            img: (np.ndarray) input image
        """
        return img[np.random.choice(len(img), self.K, replace=False), :]
        
    def _get_nearest_means(self, img, means):
        """Assigns the nearest mean label to each pixel in the input image

        Args:
            img: (np.ndarray) input image
        """
        num_pixels = len(img)
        labels = np.zeros(num_pixels)
        for i in range(num_pixels):

            # find the min squared-euclidian distance of current pixel to every mean
            dists = np.linalg.norm(img[i] - means, axis=1)
            labels[i] = np.argmin(dists)

        return labels

    def _get_centroids(self, img, labels):
        """Computes the centroids of the clustered pixels

        Args:
            img: (np.ndarray) input image
            labels: (np.ndarray) index array assigning a color to each pixel
        """
        return np.array([img[labels == k].mean(axis=0) for k in range(self.K)])

    def cluster(self, img_filename, iterations, show=False):
        """Finds the optimal pixel clusters with the EM algorithm

        Args:
            img_filename: (str) path to image file
            iterations: (int) number of iterations to run
            save: (bool) save the quantized image to file, show image if false
        """
        # read in the image
        img, im_size = self._load_image(img_filename)    

        # get the initial centroids
        centroids = self._init_means(img)

        for _ in tqdm(range(iterations), desc="Clustering pixels"):

            # E-step: assign each image pixel to its nearest centroid
            labels = self._get_nearest_means(img, centroids)

            # M-step: compute and update the centroids
            new_centroids = self._get_centroids(img, labels)
            centroids = new_centroids

        # reconstruct the image
        image_reconstructed = centroids[labels.astype(np.uint8), :]
        image = self._image_from_numpy(image_reconstructed, im_size)

        # show reconstructed image
        if show:
            image.show()
        
        # save the compressed image with '_compressed' appended to the name
        im_name = img_filename.split('.')
        cimg_filename = im_name[0] + "_compressed." + im_name[-1]
        image.save(cimg_filename)

        # print out the compression rate
        init_size = os.path.getsize(img_filename)
        final_size = os.path.getsize(cimg_filename)
        compression_rate = (init_size - final_size) / init_size
        print("Image was compressed {:.3f}%".format(100 * compression_rate))


def main():

    parser = argparse.ArgumentParser("Image compression using K-means")
    parser.add_argument("--colors", type=int, help="Number of quantized colors")
    parser.add_argument("--image-path", type=str, help="Path to image to compress")
    parser.add_argument("--show", action='store_true', 
                        help="Show the compressed image")
    parser.add_argument("--iters", type=int, 
                        help="Number of iterations to run the algorithm")
    args = parser.parse_args()

    kmeans = KMeans(args.colors)
    kmeans.cluster(args.image_path, iterations=args.iters, show=args.show)
    
    
if __name__ == '__main__':
    main()
