"""plot.py

Utility plotting functions

J. Metz <metz.jp@gmail.com>
"""

import matplotlib.pyplot as plt
from matplotlib import animation


def view_stack(data):
    """
    Wrapper around matplotlib to create a simple data viewer
    """

    fig = plt.figure()
    axis = fig.add_subplot(111)
    img = axis.imshow(data[0], cmap='gray')
    img.TMAX = len(data)
    img.tnow = 0
    img.pause = False
    title = plt.title("Frame %d" % img.tnow)

    def prevframe():
        img.tnow = (img.tnow - 1) % img.TMAX
        img.set_data(data[img.tnow])
        title.set_text("Frame %d" % img.tnow)
        fig.canvas.draw()

    def nextframe(stuff=None):
        if img.pause and (stuff is not None):
            return
        img.tnow = (img.tnow + 1) % img.TMAX
        img.set_data(data[img.tnow])
        title.set_text("Frame %d" % img.tnow)
        fig.canvas.draw()

    def press(event):
        if event.key == "left":
            prevframe()
        elif event.key == "right":
            nextframe()
        elif event.key == " ":
            img.pause ^= True
        else:
            print("Unbound key pressed:", event.key)

    fig.canvas.mpl_connect('key_press_event', press)
    animation.FuncAnimation(
        fig, nextframe, blit=False, interval=10, repeat=True)
    plt.show()
