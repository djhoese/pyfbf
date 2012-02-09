from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import figure

import numpy as np
import ifg.plot.widgets as wdgt
from keoni.fbf import Workspace

def inset(l,b,w,h,r):
    return l+w*r, b+h*r*(w/h), w-w*r*2, h-h*r*2*(w/h)

TB = wdgt.IFGToolbox(title="Toolbox")

def compare(name, x, y1,y2, records, filter=None, initial_r = 0):
    """ build a GUI with dual plots above and delta plot below
    add a slider to allow live scrolling through entire dataset
    """
    #f = Figure()
    #c = FigureCanvas(f)
    f = figure()
    c = f.canvas
    ax = f.add_axes(inset(0.0, 0.5, 1.0, 0.5, 0.05))
    dax = f.add_axes(inset(0.0, 0.0, 1.0, 0.5, 0.05))

    r0 = records[initial_r]
    filter = (lambda x: x) if (filter is None) else filter
    (l1,) = ax.plot(x, filter(y1[r0]), 'b')
    (l2,) = ax.plot(x, filter(y2[r0]), 'g')
    (ld,) = dax.plot(x, abs(filter(y2[r0])-filter(y1[r0])), 'b')

    dax.grid()
    ax.grid()
    ax.autoscale(True)
    dax.autoscale(True)
    ax.set_title(name)
    dax.set_title('[%d]' % r0)

    def refresh(r, l1=l1, l2=l2, ld=ld, x=x, y1=y1, y2=y2, c=c, ax=ax, dax=dax):
        l1.set_ydata(filter(y1[r]))
        l2.set_ydata(filter(y2[r]))
        ld.set_ydata(abs(filter(y2[r]) - filter(y1[r])))
        # FUTURE: is there a preferred methodology for rescaling?
        ax.relim()
        ax.autoscale(True, 'y')
        dax.relim()
        dax.autoscale(True, 'y')
        dax.set_title('[%d]' % r)
        c.draw()

    #nav = NavigationToolbar(c, None)
    slider = wdgt.IFGSlider(records, init_val=0, parent=TB, title=name, live_update=True, callbacks=[refresh])

    #TB.add_widget(nav)
    TB.add_widget(slider)

    c.show() # to do initial window open
    #TB.show() # FUTURE: adding nav bar seems to cause widget to hide

    return f


def link_recs(records, yfunc, canvas=None, axes=None, name="Record", autoscale=True):
    """bind a slider to an axis, updating lines using a callable
    e.g.
    plot(W.Wavenumber[0], W.RAD[0])
    recs = find(abs(W.sceneMirrorAngle[:]-180)<2.0)
    link_recs(recs, W.RAD, name="Nadir Views")
    """
    if canvas is None and axes is None:
        from matplotlib.pyplot import gcf, gca
        canvas = gcf().canvas
        axes = gca()

    if not callable(yfunc):
        ydata = yfunc
        yfunc = lambda r: ydata[r]

    def refresh(r, canvas=canvas, ax=axes, yfunc=yfunc, autoscale=autoscale):
        if len(ax.lines)==1:
            ys = [yfunc(r)]
        else:
            ys = yfunc(r)
        for y,ln in zip(ys, ax.lines):
            if y is None: continue
            ln.set_ydata(y)
        if autoscale:
            ax.relim()
            ax.autoscale(True,'y')
        canvas.draw()

    slider = wdgt.IFGSlider(records, parent=TB, title=name, live_update=True, callbacks=[refresh])
    TB.add_widget(slider)
    refresh(records[0])
    return slider


def plot(name, x,y, records, initial_r = 0):
    f = figure()
    c = f.canvas
    ax = f.add_axes(inset(0.0, 0.0, 1.0, 1.0, 0.05))

    r0 = records[initial_r]
    if callable(y): y0 = y(r0)
    else: y0 = y[r0]
    (ln,) = ax.plot(x, y0, 'k')

    ax.grid()
    ax.autoscale(True)
    ax.set_title('%s[%d]' % (name,r0))

    def refresh(r, ln=ln, x=x, y=y, c=c, ax=ax, name=name):
        if callable(y):
            ln.set_ydata(y(r))
        else:
            ln.set_ydata(y[r])
        ax.relim()
        ax.autoscale(True, 'y')
        ax.set_title('%s[%d]' % (name,r))
        c.draw()

    #nav = NavigationToolbar(c, None)
    slider = wdgt.IFGSlider(records, init_val=0, parent=TB, title=name, live_update=True, callbacks=[refresh])

    #TB.add_widget(nav)
    TB.add_widget(slider)

    c.show() # to do initial window open
    #TB.show() # FUTURE: adding nav bar seems to cause widget to hide

    return f

def browse(x, y, r=None):
    from matplotlib.pyplot import figure, plot, grid
    if r is None:
        r = np.arange(len(y))
    figure()
    plot(x, y[r[0]])
    grid()
    link_recs(r, y)




# run in ipython --pylab=qt
def test1(Wdir=None, Xdir=None, name='BT', wnum_name='Wavenumber'):
    W = Workspace(Wdir or '/Volumes/data/tmp/sh110909p3o4pr110910')
    X = Workspace(Xdir or '/Volumes/data/tmp/sh110909p3o4pr110920')
    wnum = getattr(X,wnum_name)[0]
    a = getattr(W, name)
    b = getattr(X, name)
    nW = len(a)
    nX = len(b)
    assert(nX==nW)
    recs = np.arange(nW, dtype=int)
    f = compare(name, wnum, a, b, recs)
    return W,X,f
