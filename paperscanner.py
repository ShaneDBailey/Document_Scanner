"""
Author: Shane Bailey

Program Description:
- This program serves to take in a photograph of a document, 
- and transform/position the document so that it is upright 

Library Resources:
- pip install numpy
- pip install opencv-python

Important References:
- 

"""
import cv2
import numpy

"""
Serves to rotate a given rectangle portion of an image to be upright and take up the full image
- Detemines the directional location of each point given
- Finds the width and height of the rectangle
- Sets the destination of the transform points to be
- Performs the perspective transform
- Returns the translated image
"""
def four_point_perspective_transform(image, points):
    #We can find which point is where in reference to top/bottom right/left, 
    rectangle = numpy.zeros((4, 2), dtype="float32")
    sums_of_coords = points.sum(axis=1)#(x+y)lowest sum will be closest to 0,0 or topleft, and the opposite the bottom right
    difference_of_coords = numpy.diff(points, axis=1)#(y-x)lowest differnce will be cloest to top right and opposite bottom-left
    rectangle[0] = points[numpy.argmin(sums_of_coords)]  # top-left
    rectangle[2] = points[numpy.argmax(sums_of_coords)]  # bottom-right
    rectangle[1] = points[numpy.argmin(difference_of_coords)]  # top-right
    rectangle[3] = points[numpy.argmax(difference_of_coords)]  # bottom-left

    #Find the height and width of the rectangle, linalg is used to find the hyptonuse
    widthA = numpy.linalg.norm(rectangle[2] - rectangle[3])
    widthB = numpy.linalg.norm(rectangle[1] - rectangle[0])
    maxWidth = max(int(widthA), int(widthB))
    heightA = numpy.linalg.norm(rectangle[1] - rectangle[2])
    heightB = numpy.linalg.norm(rectangle[0] - rectangle[3])
    maxHeight = max(int(heightA), int(heightB))

    # Set the destination points
    destination = numpy.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Perform the perspective transform
    Map = cv2.getPerspectiveTransform(rectangle, destination)
    warped = cv2.warpPerspective(image, Map, (maxWidth, maxHeight))#map the 

    return warped

class Document_Scanner:
    def __init__(self, image):
        self.original_document = image
        self.processed_image = image.copy()
        self.warped_image = None
        self.document_detection()

    def document_detection(self):
        gray_filter = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)#gray scale
        edges = cv2.Canny(gray_filter, 100, 300)#line dectection
        
        kernel = numpy.ones((5, 5), numpy.uint8)#define size to dilate at
        dilated = cv2.dilate(edges, kernel, iterations=1)#thicken lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#find contours

        #Find the biggest contour
        max_area = 0
        biggest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                biggest_contour = contour

        if biggest_contour is not None:
            #approximate the contour to a rectangle
            perimeter = cv2.arcLength(biggest_contour, True)
            approx = cv2.approxPolyDP(biggest_contour, 0.01 * perimeter, True)
            #check if its a rectangle
            if len(approx) == 4:
                warped = four_point_perspective_transform(self.processed_image, approx.reshape(4, 2))
                cv2.imwrite("document_scan.jpg", warped)#save the scan
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("gray_scaled.jpg", warped)#save a grayscale of the scan
                _,warped=cv2.threshold(warped,123,255,cv2.THRESH_BINARY)
                cv2.imwrite("school_scan.jpg", warped)#save a black white threshold of the scan
                self.warped_image = warped

# Example usage read in the image
image = cv2.imread('receipt.jpg')
document = Document_Scanner(image)
cv2.imshow("Contoured_image", document.warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
