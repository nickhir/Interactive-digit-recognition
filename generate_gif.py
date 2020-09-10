from PIL import Image
import os
import numpy as np

frames = []
images = os.listdir("images")
for im in images:
    new_frame = Image.open(f"images/{im}", mode="r")
    np_image = np.array(new_frame)
    image = Image.fromarray(np_image)

    frames.append(image)

print(frames)
frames[0].save('animated.gif',
               save_all=True, append_images=frames[1:], optimize=False, duration=300, loop=0)