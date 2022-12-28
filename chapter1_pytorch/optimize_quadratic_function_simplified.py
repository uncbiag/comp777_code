from comp777_code.optimize_fcn import optimize_fcn_with_torch
import torch
from bokeh.plotting import show
import holoviews as hv
from holoviews import opts

#define the function we want to optimize, add dummy opts
def f(x, opts=None):
    return (x-2)**2

# initialize the variable we want to optimize; make sure we can compute the gradient
x = torch.tensor([-1.0], requires_grad=True)

# now create a PyTorch optimizer and tell it to optimize over x
optimizer = torch.optim.SGD([x], lr=0.025, momentum=0.9 )
iterations = 100

# call the boilerplate optimization code
x_star, history = optimize_fcn_with_torch(x, fcn=f, fcn_pars=None,
                                          nr_of_iterations=100, optimizer=optimizer,
                                          record_history=True, display_loss=True)

# and show the result, including a horizontal line for the correct solution
horizontal_line = hv.HLine(2.0).opts(opts.HLine(color='red', line_width=2),)
history_plot = hv.Curve(history).opts(xlabel=("iterations"))
show(hv.render(history_plot*horizontal_line))
