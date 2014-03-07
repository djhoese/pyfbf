from matplotlib.pyplot import figure

import numpy as np
import ifg.plot.widgets as wdgt
from .workspace import Workspace

# FIXME: Don't magically use globals. Especially ones that require QtApplications
TB = None

def inset(l,b,w,h,r):
    return l+w*r, b+h*r*(w/h), w-w*r*2, h-h*r*2*(w/h)

def create_base_window():
    global TB
    TB = wdgt.IFGToolbox(title="Toolbox")
    return TB

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


def _iterable(x): return hasattr(x,'__iter__')


def follow(records, yfuncs, axes=None, canvas=None, name="Record", callbacks=[], autoscale=True):
    """bind a slider to an axis, updating lines using a callable
    e.g.
    from keoni.fbf import Workspace
    W = Workspace('.')
    import keoni.fbf.plots as gui
    reim = lambda re,im,n: abs(re[n] + 1j*im[n])
    fhy = lambda r: abs(W.CalHotRefLWReal[r] + 1j* W.CalHotRefLWImag[r])
    fsy = lambda r: abs(W.CalSceneLWReal[r] + 1j* W.CalSceneLWImag[r])
    fcy = lambda r: abs(W.CalColdRefLWReal[r] + 1j* W.CalColdRefLWImag[r])
    figure()
    subplot(311); pah=plot(fhy(0))
    subplot(312); pas=plot(fsy(0))
    subplot(313); pac=plot(fcy(0))    
    gui.follow(list(range(len(W.CalSceneLWReal))), [fhy,fsy,fcy])
    """
    from matplotlib.pyplot import gcf, gca
    if not _iterable(yfuncs):
        yfuncs = [yfuncs]
    if axes is not None and not _iterable(axes):
        axes = [axes]
    if axes is None:
        gcax = list(gcf().axes)
        if len(yfuncs)==1:
            axes = [gca()]            
        elif len(yfuncs)==len(gcax):
            axes = gcax
        else:
            raise ValueError('could not figure out which axes to use')
    if canvas is None:
        canvas = gcf().canvas
    if not _iterable(callbacks):
        callbacks = [callbacks]

    def refresh(r, canvas=canvas, ax=axes, yfuncs=yfuncs, autoscale=autoscale, callbacks=callbacks):
        for fy,a in zip(yfuncs, ax):
            ydata = fy(r if callable(fy) else fy[r])
            if (len(a.lines)==1):
                a.lines[0].set_ydata(ydata)
            else:
                for ln,y in zip(a.lines, ydata):
                    ln.set_ydata(y)
            if autoscale:
                a.relim()
                a.autoscale(True, 'y')
        for cb in callbacks:
            cb(r)
        canvas.draw()

    slider = wdgt.IFGSlider(records, parent=TB, title=name, live_update=True, callbacks=[refresh])
    TB.add_widget(slider)
    refresh(records[0])
    return slider, refresh


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
