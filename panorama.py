import cv2
import numpy as np


# Detect keypoints in the provided image and extract local invariant descriptors
# Returns the keypoints and features
def detectAndDescribe(image):
    # create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # detectAndCompute first find keypoints, then computes the descriptors
    # can pass in a mask to section of part of image, but here mask is None
    (kps,descriptors) = sift.detectAndCompute(image,None)

    # convert keypoints from KeyPoint objects to numpy arrays
    kps = np.float32([kp.pt for kp in kps])

    return kps,descriptors

# To match features: loop over the descriptors from both images, compute the distances,
# and find the smallest distance for each pair of descriptors
def matchKeypoints(kpsA,kpsB,featuresA,featuresB,ratio,reprojThresh):
    # cv2.DescriptorMatcher_create function is an object that matches the descriptors
    # Takes the descriptor of one feature in first set and is matched with all other features in
    # second set using some distance calculation. And the closest one is returned
    # BruteForce exhaustively computes the Euclidean distance between all feature vectors from both images
    # and finds the pairs of descriptors that have the smallest distance
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # cv2.drawMatchesKnn draws all the k best matches between the two feature vector sets
    # If k=2, it will draw two match-lines for each keypoint. We take top two matches so we
    # can apply David Lowe's ratio test for false-positive match pruning
    rawMatches = matcher.knnMatch(featuresA,featuresB,2)
    matches = []

    # Some of the rawMatches found may be false positives meaning the feature found in each image don't
    # actually match each other. We can prune these false positives by using the David Lowe's ratio test
    # This test determines high quality feature matches by looping over all matches, and taking a ratio
    # of distance from the closest neighbor to the distance of the second closest.
    # Typical values for Lowe’s ratio are normally in the range [0.7, 0.8].
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # compute homography which requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        # we want to find the transformation matrix between the each keypoint from A and B so that we know
        # how to transform the second image to align with the first image
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)

        # return the matches, the homograpy matrix and status of each matched point
        return (matches, H, status)

        # otherwise, no homograpy could be computed
    return None


# Visualize keypoint matches between the two images
def drawMatches(imageA,imageB,kpsA,kpsB,matches,status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    # trainIdz is from our first image, queryIdx is from our second image
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis

# image order is important, expecting left to right
# ratio is used for David Lowe's ratio test when matching features
# reprojThresh is the  max pixel wiggle room allowed by RANSAC
# showMatches is a bool to determine if you keypoint matches should be shown or not
# Returns the panorama result
def stitch(images, ratio=0.75,reprojThresh=4.0,showMatches=False):
    (imageB,imageA) = images

    # Detect the keypoints and extracts local invariant descriptors (SIFT) from both images
    (kpsA,featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)

    # Now that we have the keypoints and descriptors (features), we can match the features in the two images
    M = matchKeypoints(kpsA,kpsB,featuresA,featuresB,ratio,reprojThresh)

    # if there are not enough matched keypoints to build the panorama, then return
    if M is None:
        return None

    # if we have enough matched keypoints, apply perspective transform
    # matches is a list of keypoint matches
    # H is the homography matrix we'll use in the perspective transform, H is created by RANSAC algo
    # status is a list of indexes to indicate which keypoints in matches were successfully spatially verified by RANSAC
    (matches, H, status) = M

    # Apply perspective transform to second image
    # first argument is the image we want to warp (our right image)
    # second argument is the 3x3 homography matrix
    # third argument is the shape of the output image (same height, but double the width since stitching side by side)
    result = cv2.warpPerspective(imageA,H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # Visualize keypoint matches if indicated
    if showMatches:
        vis = drawMatches(imageA,imageB,kpsA,kpsB,matches,status)
        return (result,vis)

    return result



imageA = cv2.imread("images/panorama1.jpg")
imageB = cv2.imread("images/panorama2.jpg")

(result,vis) = stitch([imageA,imageB], showMatches=True)
# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)



# Notes:
# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html

# Keypoints are spatial locations in an image that define something that is interesting. These keypoints are
#     invariant to rotation, translation, and scale (shrinking/expanding), and invariant to distortion (projective
#     transformation or homography). This means no matter how the image changes, you should be able to find
#     the same keypoints.
# Descriptors are what describe the keypoints. Each keypoint has a descriptor. Descriptors describe both scale
#     and orientation of the keypoint. This is needed if matching keypoints between images
# A keypoint in an image is shown as a circle with varying radius and a line from the center to the circle edge.
#   Scale: An image is decomposed into multiple scales, then either LoG or DoG is used to find potential keypoints
#      at different scales. We compare a point with it's surrounding 8 pixel neighbors as well as the 9 pixels in
#      previous scale and the next scale. This is basically searching the image scales for a local extrema. It means
#      that the keypoint is best represented in that particular scale. Since we detect keypoints at different scales,
#      the radius of the circle represents at what scale the feature was best detected.
#   Orientation: The algorithm searchs a pixel neighbourhood that surrounds the keypoint and figures out how this
#      pixel neighbourhood is oriented. It detects the most dominant orientation of the gradient angles in the
#      neighborhood. We can determine if a feature in one image is the same as in another rotated image by looking
#      at the way the neighbors of the rotated keypoint pixel are oriented.


# Both SIFT (Scale Invariant Feature Transform) and SURF (Speeded Up Robust Features) detect and describe keypoints.

# SIFT looks at not only x,y of keypoints but scale as well.
# The process for finding SIFT keypoints is:
#    1) Blur and resample the image with different blur widths and sampling rates to create a scale space
#    2) Use the difference of gaussians method to detect blobs at different scales; the blob centers become our
#       keypoints at a given x, y, and scale
#    3) Assign every keypoint an orientation by calculating a histogram of gradient orientations for every pixel
#       in its neighbourhood and picking the orientation bin with the highest number of counts
#    4) Assign every keypoint a 128-dimensional feature vector based on the gradient orientations of pixels in
#       16 local neighbourhoods

# SURF accomplishes the same goals as SIFT, but uses tricks to increase the speed. It uses the determinant of Hessian
# to find blobs (the center of the blobs will be the keypoints, same in SIFT). Dominant orientation found by
# examining the horizontal and vertical response to Harr wavelets. And then resulting vector is 64 dimensions, not 128



