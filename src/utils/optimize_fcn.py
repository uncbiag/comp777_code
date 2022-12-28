import torch
import holoviews as hv
hv.extension('bokeh')
from bokeh.io import export_svgs

def export_svg(obj, filename):
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'svg'
    export_svgs(plot_state, filename=filename)

def optimize_fcn_with_torch(x, fcn, fcn_pars=None, nr_of_iterations=100, optimizer=None, record_history=False, display_loss=True):

    if optimizer is None:
        optimizer = torch.optim.SGD([x], lr=0.025, momentum=0.9)

    history = []

    for i in range(nr_of_iterations):

        if display_loss:
            print('Iter {}: x={}'.format(i,x.item()))

        y = fcn(x,fcn_pars)
        if record_history:
            history.append( (float(i),x.item()))

        # backprop
        optimizer.zero_grad()
        y.backward()
        optimizer.step()

    return x, history