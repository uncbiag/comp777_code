from comp777_code.optimize_fcn import optimize_fcn_with_torch
from comp777_code.optimize_fcn import optimize_image_with_torch

from comp777_code import finite_differences as FD
import torch
from bokeh.plotting import show
import holoviews as hv
from holoviews import opts

spacing = torch.tensor([1.0])
fd = FD.FD(spacing=spacing)

#define the function we want to optimize, add dummy opts
def f(x, opts=None):

    tst = fd.lap(x,bc=FD.BoundaryCondition.NEUMANN_ZERO)

    #return (fd.lap(x,bc=FD.BoundaryCondition.DIRICHLET_ZERO)**2).sum()
    return (fd.lap(x,bc=FD.BoundaryCondition.NEUMANN_ZERO)**2).sum()

# initialize the variable we want to optimize; make sure we can compute the gradient
nr_of_pts = 25
x0 = torch.zeros([1,1,nr_of_pts])

x0[0,0,nr_of_pts//4:3*nr_of_pts//4] = 1

#x0[0,0,10:20] = 1
#x0[0,0,30:40] = -1
#x0[0,0,50:60] = 1
#x0[0,0,70:80] = -1


x0_np = (x0[0,0,...]).clone().detach().numpy()

x = x0
x.requires_grad = True

# now create a PyTorch optimizer and tell it to optimize over x
optimizer = torch.optim.SGD([x], lr=0.02, momentum=0.9 )
iterations = 20000

# call the boilerplate optimization code
x_star, history = optimize_image_with_torch(x, fcn=f, fcn_pars=None,
                                          nr_of_iterations=iterations, optimizer=optimizer,
                                          record_history=True, display_loss=True)

x_star_np = (x_star[0,0,...]).detach().numpy()

x0_to_plot = [(i,x0_np[i]) for i in range(nr_of_pts)]
x_star_to_plot = [(i,x_star_np[i]) for i in range(nr_of_pts)]

# and show the result, including a horizontal line for the correct solution
#horizontal_line = hv.HLine(2.0).opts(opts.HLine(color='red', line_width=2),)
x0_plot = hv.Curve(x0_to_plot).opts(xlabel=('x'),color='red')
x_star_plot = hv.Curve(x_star_to_plot).opts(color='blue')

show(hv.render(x0_plot*x_star_plot))
