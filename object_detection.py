import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

#image=cv2.imread('image/computer.jpg')
#image=cv2.imread('image/bike.jpeg')
#image=cv2.imread('image/car.jpg')
#image=cv2.imread('image/cup.jpg')
#image=cv2.imread('image/bottle.jpg')
#image=cv2.imread('image/cat.jpg')
#image=cv2.imread('image/fruit.png')
image=cv2.imread('image/dog-and-cat-cover.jpg')
    
bbox, label, conf = cv.detect_common_objects(image)

output_image = draw_bbox(image, bbox, label, conf)

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 800, 800)

cv2.imshow('Detection', output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# from imageai.Detection import ObjectDetection
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath("resnet50_imagenet_tf.2.0.h5")

# detector.loadModel()

# detections = detector.detectObjects
# FromImage(input_image="image/dog-and-cat-cover.jpg", output_image_path="bike.jpg", minimum_percentage_probability=30)
# print(detections)