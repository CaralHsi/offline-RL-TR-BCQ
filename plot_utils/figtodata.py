import numpy as np
import PIL.Image as Image


def fig2data(fig):
  """
  fig = plt.figure()
  image = fig2data(fig)
  @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
  @param fig a matplotlib figure
  @return a numpy 3D array of RGBA values
  """
  # draw the renderer
  fig.canvas.draw()

  # Get the RGBA buffer from the figure
  w, h = fig.canvas.get_width_height()
  buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (w, h, 4)

  # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
  buf = np.roll(buf, 3, axis=2)
  image = Image.frombytes("RGBA", (w, h), buf.tostring())
  image = np.asarray(image)
  return image