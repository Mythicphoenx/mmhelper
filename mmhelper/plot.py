#
# FILE        : plot.py
# CREATED     : 22/09/16 13:24:16
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Main plotting functions
#

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def view_stack(data):
    """
    Wrapper around matplotlib to create a simple data viewer
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data[0], cmap='gray')
    im.TMAX = len(data)
    im.tnow = 0
    im.pause = False
    title = plt.title("Frame %d" % im.tnow)

    def prevframe():
        im.tnow = (im.tnow - 1) % im.TMAX
        im.set_data(data[im.tnow])
        title.set_text("Frame %d" % im.tnow)
        fig.canvas.draw()

    def nextframe(stuff=None):
        if im.pause and (stuff is not None):
            return
        im.tnow = (im.tnow + 1) % im.TMAX
        im.set_data(data[im.tnow])
        title.set_text("Frame %d" % im.tnow)
        fig.canvas.draw()

    def press(event):
        if event.key == "left":
            prevframe()
        elif event.key == "right":
            nextframe()
        elif event.key == " ":
            im.pause ^= True
        else:
            print("Unbound key pressed:", event.key)

    fig.canvas.mpl_connect('key_press_event', press)
    ani = animation.FuncAnimation(
        fig, nextframe, blit=False, interval=10, repeat=True)

    plt.show()
