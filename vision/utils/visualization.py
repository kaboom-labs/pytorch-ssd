from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def plot_image_grid(images: list):
    '''
    Plot a grid of images.
    '''
    fig = plt.figure(figsize=(1.,1.))
    grid = ImageGrid(fig, 111, 
            nrows_ncols=(1,3),
            axes_pad=0.1,
            )
    for ax, im in zip(grid, images):
        ax.imshow(im)

    plt.show()

def make_square(img, background_color, square_size):
    sq_img = None
    img = Image.fromarray(img)

    width, height = img.size
    if width == height:
        sq_img = img
    elif width > height:
        result = Image.new(img.mode, (width,width), background_color)
        result.paste(img, (0, (width-height) //2))
        sq_img = result
    else:
        result = Image.new(img.mode, (height, height), background_color)
        result.paste(img, ((height-width) //2, 0))
        sq_img = result
    return sq_img.resize((square_size, square_size))


