import torch
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from bokeh.plotting import show
from bokeh.io import export_svgs

def export_svg(obj, filename):
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'svg'
    export_svgs(plot_state, filename=filename)

def f(x):
    return (x-2)**2

# first compute the function over the interval [-2,4]
x = torch.linspace(-2,4,100)
y = f(x)

# now plot it with holoviews and save it out as a file
pts = list(zip(x.numpy(),y.numpy()))
fcn_plot = hv.Curve(pts)
export_svg(fcn_plot, 'ch1_fcn_plot.svg')
show(hv.render(fcn_plot))

# now numerically find the minimum, initialzing at -1
x = torch.tensor([-1.0], requires_grad=True)

# now create a PyTorch optimizer and tell it to optimize over x
optimizer = torch.optim.SGD([x], lr=0.025, momentum=0.9 )
iterations = 100

history = []

# iterate a few times
for i in range(iterations):

    print('Iter {}: x={}'.format(i,x.item()))

    y = f(x)
    history.append( (float(i),x.item()))

    # backprop
    optimizer.zero_grad()
    y.backward()
    optimizer.step()

# and show the result, including a horizontal line for the correct solution
horizontal_line = hv.HLine(2.0).opts(opts.HLine(color='red', line_width=2),)
history_plot = hv.Curve(history).opts(xlabel=("iterations"))
export_svg(history_plot*horizontal_line, 'ch1_history_plot.svg')
show(hv.render(history_plot*horizontal_line))
