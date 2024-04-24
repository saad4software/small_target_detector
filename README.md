# small_target_detector

small target detector algorithm that utilize dense optical flow for target detection and uses k-means for target recognition. a special tracker was also proposed to overcome optical flow disadvantages regarding stationary targets

## proposed algorithms
- k-means based algorithm, uses only dense optical flow and k-means without any ML implementation
- cnn single: uses a special network archetecture to filter optical flow proposed areas
- cnn pairs: uses information from both optical flow frames to filter possible targets from background noise
- k-means only: uses k-means for both detection and recogntion, it has hight compututional cost and still an idea
