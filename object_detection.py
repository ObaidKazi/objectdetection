import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO
image=cv2.imread('image/computer.jpg')
#image=cv2.imread('image/bike.jpeg')
#image=cv2.imread('image/car.jpg')
#image=cv2.imread('image/cup.jpg')
#image=cv2.imread('image/bottle.jpg')
#image=cv2.imread('image/cat.jpg')
#image=cv2.imread('image/fruit.png')
#image=cv2.imread('image/dog-and-cat-cover.jpg')
    
#bbox, label, conf = cv.detect_common_objects(image,confidence=0.7)
# output_image = draw_bbox(image, bbox, label, conf)

# cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Detection', 800, 800)

# cv2.imshow('Detection', output_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#other way to detect object 

model = YOLO("yolov8x.pt")

results=model.predict(source=image,show=True) 
names = model.names

for r in results:
    for c in r.boxes.cls:
        print(names[int(c)])

while True:
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break