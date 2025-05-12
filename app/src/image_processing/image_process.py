# Import libraries
import cv2
import numpy as np



#//==================================//
class image_processing:
    """
    A class for performing various image processing operations.
    """

    def __init__(self, image):
        """
        Initialize the image_processing class with an image.
        """
        self.image = image
        # print('=== image_processing_init ====')
        # self.image_resize()

    #//==================================//
    def image_resize(self, scale):
        """
        Resize the image by a given scale percentage.
        """
        # print('=== image_resize ===')
        # print('image :=', type(self.image))
        # print('shape := ', self.image.shape)
        # print('size := ', self.image.size)
        # print('dtype  := ', self.image.dtype)

        # Calculate the new dimensions based on the scale percentage
        scale_percent = scale
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize the image using INTER_AREA interpolation
        image_resize = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)

        # Uncomment the following lines to display the original and resized images
        # cv2.namedWindow("img_original")
        # cv2.moveWindow("img_original", 250, 250)
        # cv2.imshow('img_original', self.image)

        # cv2.namedWindow("img_resize")
        # cv2.moveWindow("img_resize", 250, 250)
        # cv2.imshow('img_resize', self.image)
        # cv2.waitKey()

        return image_resize

    # //=======================================//
    def image_rgb2bgr(self):
        """
        Convert the image from RGB to BGR color space.
        """
        print('=== image_color_conversion ===')
        self.image = cv2.cvtColor(np.array(self.image),cv2.COLOR_RGB2BGR)
        cv2.imshow('img_bgr', self.image)

    # //=======================================//
    def image_grayscale(self):
        """
        Convert the image to grayscale if it is in color.
        """
        print('=== image_grayscale ===')
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:     # Check if the image is in color
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img_gray', self.image)

    # //=======================================//
    def image_gray2rgb(self):
        """
        Convert a grayscale image back to RGB color space.
        """
        print('=== image_gray2rgb ===')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        cv2.imshow('img_rgb', self.image)

    # //=======================================//
    def image_threshold1(self):
        """
        Apply binary thresholding with Otsu's method to the image.
        """
        print('=== image_threshold1 ===')
        self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imshow('img_thres1', self.image)

    # //=======================================//
    def image_threshold2(self):
        """
        Apply inverse binary thresholding with Otsu's method to the image.
        """
        print('=== image_threshold2 ===')
        self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imshow('img_thres2', self.image)

    # //=======================================//
    def image_bgr2gray(self):
        """
        Convert the image from BGR to grayscale.
        """
        print('=== image_bgr2gray ===')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray', self.image)

    # //=======================================//
    def image_blur(self):
        """
        Apply Gaussian blur to the image to reduce noise.
        """
        print('=== image_blur ===')
        self.image = cv2.GaussianBlur(self.image, (3,3), 0)
        cv2.imshow('img_blur', self.image)
