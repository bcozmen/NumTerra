import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2

sp = ax.streamplot(X, Y, U, V, broken_streamlines=False)
print("sp structure:", type(sp))
print("lines:", hasattr(sp, 'lines'))
print("arrows:", hasattr(sp, 'arrows'))

try:
    sp.lines.remove()
    print("removed lines")
except Exception as e:
    print("err removing lines:", e)

try:
    # try patching _remove_contours
    class DummyStream:
        def __init__(self, sp):
            self.lines = sp.lines
            self.arrows = getattr(sp, 'arrows', None)
        def remove(self):
            self.lines.remove()
            if self.arrows:
                self.arrows.remove()

    DummyStream(sp).remove()
    print("removed via DummyStream")
except Exception as e:
    print("err DummyStream:", e)
