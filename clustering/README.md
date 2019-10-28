# Clustering
An introduction to unsupervised learning and clustering. We will cover:
1. The K-means and Expectation Maximization (EM) algorithm
2. An application of the K-means algorithm for image compression

## Usage
To compress an image run the `k_means.py` script with command line arguments:
```
  -h, --help            show this help message and exit
  --colors COLORS       Number of quantized colors
  --image-path IMAGE_PATH
                        Path to image to compress
  --show                Show the compressed image
  --iters ITERS         Number of iterations to run the algorithm
```

+ Example: `python3 k_means.py --image-path=images/place.png --colors=64 --iters=20 --show`

+ You can compare the file sizes of the compressed and uncompressed image with `ls -lh images/`
