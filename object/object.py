import cv2
import numpy as np
import random

def random_color():
    """ Generate random color for drawing shapes """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class Shape:
    def __init__(self, frame):
        self.color = random_color()
        self.draw(frame)
        self.name = self.__class__.__name__
        self.location = None

    def draw(self, frame):
        pass

class Circle(Shape):
    def __init__(self, frame):
        self.center = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
        self.radius = random.randint(10, 100)
        super().__init__(frame)

    def draw(self, frame):
        cv2.circle(frame, self.center, self.radius, self.color, -1)

    def get_x_y_r(self):
        return self.center[0], self.center[1], self.radius
    
    def get_center(self):
        return self.center



class Rectangle(Shape):
    def __init__(self, frame):
        self.pt1 = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
        self.pt2 = (random.randint(self.pt1[0], frame.shape[1]), random.randint(self.pt1[1], frame.shape[0]))
        super().__init__(frame)

    def draw(self, frame):
        cv2.rectangle(frame, self.pt1, self.pt2, self.color, -1)

    def get_x_y_w_h(self):
        return self.pt1[0], self.pt1[1], self.pt2[0] - self.pt1[0], self.pt2[1] - self.pt1[1]
    
    def get_center(self):
        return (self.pt1[0] + self.pt2[0])//2, (self.pt1[1] + self.pt2[1])//2
    
    def get_width(self):
        return self.pt2[0] - self.pt1[0]
    
    def get_height(self):
        return self.pt2[1] - self.pt1[1]
    

class Triangle(Shape):
    def __init__(self, frame):
        self.pt1 = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
        self.pt2 = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
        self.pt3 = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
        super().__init__(frame)

    def draw(self, frame):
        triangle_cnt = np.array([self.pt1, self.pt2, self.pt3])
        cv2.drawContours(frame, [triangle_cnt], 0, self.color, -1)

    def get_x_y(self):
        return self.pt1[0], self.pt1[1], self.pt2[0], self.pt2[1], self.pt3[0], self.pt3[1]
    
    def get_center(self):
        return (self.pt1[0] + self.pt2[0] + self.pt3[0])//3, (self.pt1[1] + self.pt2[1] + self.pt3[1])//3
    



def main():
    NUMBER_OF_SHAPES = 5  # Define the fixed number of shapes
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Initialize shapes
    shapes = []
    ret, frame = cap.read()
    if not ret:
        print("Error: Initial frame capture failed. Exiting...")
        return

    for _ in range(NUMBER_OF_SHAPES):
        shape_type = random.choice([Circle, Rectangle, Triangle])
        shapes.append(shape_type(frame))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Draw all shapes
        for shape in shapes:
            shape.draw(frame)

        cv2.imshow('Frame with Shapes', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
