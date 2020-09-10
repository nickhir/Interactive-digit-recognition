import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pygame
import sys
from PIL import Image
import numpy as np
from tensorflow import keras

pygame.init()
WIDTH = 1000
HEIGHT = 700

DRAW_AREA = pygame.Rect(0, 0, WIDTH / 1.5, 560)
RESULTS_AREA = pygame.Rect(WIDTH / 1.5, 0, WIDTH / 3, HEIGHT)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (79, 235, 70)
RED = (219, 64, 64)
ORANGE = (209, 152, 61)
BLUE = (28, 58, 200)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BLACK)
drawing_surface = screen.subsurface(DRAW_AREA)
results_surface = screen.subsurface(RESULTS_AREA)

pygame.display.set_caption("Number Recognizer")
clock = pygame.time.Clock()

continue_loop = True
drawing = True

model = keras.models.load_model("models/MNIST_conv.h5", compile=False)


def draw_text(msg, x, y, color, size, bold=False, italic=False):
    font = pygame.font.SysFont("Arial", size, bold=bold, italic=italic)
    text = font.render(msg, False, color)
    screen.blit(text, [x, y])


def create_snapshot():
    image_string = pygame.image.tostring(drawing_surface, "RGB")
    image = Image.frombytes("RGB", (int(WIDTH / 1.5), 560), image_string)
    image = cropping(image)
    # we have to resize them because training pictures are that size
    image = image.resize((28, 28))
    #plt.imshow(image)
    #plt.show()
    # turn image into an array
    np_image = np.array(image)
    # Each channel of the RGB image has the same value, we only need one, so extract one
    np_image = np_image[:, :, 0] / 255.0

    return np_image


def cropping(image):
    # ultimately, we want to determine the border pixels of the drawn digit and then crop accordingly. This significantly
    # improves performance, because image quality doesnt get corrupted as much, and digits always in the center of image
    white_pixels = []
    # iterate over each pixel
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if image.getpixel((x, y))[0] != 0:
                white_pixels.append((x, y))

    # get the smallest and biggest x coordinate:
    x_small = sorted(white_pixels)[0][0]
    x_big = sorted(white_pixels)[-1][0]

    # get the smallest and biggest y coordinate:
    y_small = sorted(white_pixels, key=lambda x: x[1])[0][1]
    y_big = sorted(white_pixels, key=lambda x: x[1])[-1][1]

    image = image.crop((x_small - 45, y_small - 45, x_big + 45, y_big + 45))
    return image


def predict_digit(np_image):
    # we only predict a single picture, but our model was trained on a batch of pictures -> have to simulate that dimesion
    np_image = np_image.reshape((1, 28, 28, 1))
    output = model.predict(np_image)
    result = str(np.argmax(output[0]))
    confidence = np.max(output[0])
    confidence = round(confidence, 4) * 100
    return output, result, confidence


def prediction_numbers(x, y, distance):
    for i in range(10):
        draw_text(f"{i}", x, y + i * distance, WHITE, size=25)
        pygame.draw.line(screen, WHITE, (710, 60 + i * distance), [710, 80 + i * distance], 5)


while continue_loop:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:  # exit game if red X in top right corner is pressed
            sys.exit()

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            sys.exit()

    # check if left mouse is pressed down, and if in drawing area
    if pygame.mouse.get_pressed()[0] and drawing and pygame.mouse.get_pos()[0] < (WIDTH / 1.5) - 10 and \
            pygame.mouse.get_pos()[1] < 550:
        pygame.draw.circle(screen, WHITE, pygame.mouse.get_pos(), 10)

    # when user presses enter, a snapshot of the drawing area is created and saved
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RETURN:
            drawing = False
            np_image = create_snapshot()
            output, prediction, confidence = predict_digit(np_image)

            # generate prediction bars
            for i in range(10):
                pygame.draw.rect(screen, BLUE, pygame.Rect((713, 63 + i * 35), (250 * output[0][i], 14)))

            draw_text(f"{prediction}", 828, 460, RED, size=30, bold=True)
            draw_text(f"{confidence:.2f}%", 783, 555, GREEN, size=30, bold=True)

    # when user presses space, display is cleared, and new digits can be drawn:
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
            drawing = True
            drawing_surface.fill(BLACK)
            results_surface.fill(BLACK)

    # seperate screen in area where you draw the digits and one where the output is displayed
    pygame.draw.line(screen, WHITE, [(WIDTH / 1.5) + 2, 0], [(WIDTH / 1.5) + 2, HEIGHT], 3)
    pygame.draw.line(screen, WHITE, [0, 563], [(WIDTH / 1.5) + 2, 563], 3)

    # generate permanent text
    draw_text("created by nickhir", x=240, y=625, color=WHITE, size=20, italic=True)
    draw_text("Predictions", x=756, y=5, color=WHITE, size=25, bold=True)
    draw_text("Ultimate prediction", x=712, y=420, color=WHITE, size=27, bold=True)
    draw_text("Confidence", x=750, y=515, color=WHITE, size=30, bold=True)
    draw_text("press space to clear screen", x=732, y=615, color=WHITE, size=20)
    draw_text("press enter to predict", x=745, y=640, color=WHITE, size=20)
    prediction_numbers(690, 55, 35)
    clock.tick(250)
    pygame.display.update()
