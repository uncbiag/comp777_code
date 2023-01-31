"""
*finite_difference.py* is the main package to compute finite differences
on pytorch tensors. The package supports first and second order derivatives.
"""

# from builtins import object
from enum import Enum

import torch

class BoundaryBehavior(Enum):
    ZERO = 1  # set out of boundary to zero
    REFLECT = 2  # reflect it e.g., 1 0 1 2 3 ...
    COPY = 3  # copy from the closest boundary 0 0 1 2 3 ...


class BoundaryCondition(Enum):
    DIRICHLET_ZERO = 1  # Dirichlet zero boundary
    NEUMANN_ZERO = 2  # Neumann boundary conditions for finite difference operators

class FD:
    """
    *FD* is the basic finite difference class.
    All the methods expect that BxCxXxYxZ format.
    I.e., first batch, then channel, then the spatial dimensions.
    """

    def __init__(self, spacing):
        """
        Constructor
        :param spacing: 1D pytorch array defining the spatial spacing, e.g., [0.1,0.1,0.1] for a 3D image
        """

        self.spacing = spacing
        """spacing"""
        self.dim = self.spacing.numel()
        """spatial dimension"""
        self.supported_dims = [1, 2, 3]
        """supported spatial dimensions"""
        if self.dim not in self.supported_dims:
            raise ValueError(
                "Finite differences are only supported in dimensions: {}".format(
                    self.supported_dims
                )
            )

    def get_spatial_dimension(self, im):
        """
        Method to return the dimension of an input image, im, stripping the batch and channel dimensions
        :param im: Input image
        :return: Returns the dimension of the image I without the batch and channel dimensions
        """
        dim = im.dim() - 2
        if dim not in [1, 2, 3]:
            raise ValueError(
                "Finite differences are only supported in dimensions: {}".format(
                    self.supported_dims
                )
            )
        return dim

    def check_supported_dimensions(self, im):
        """
        Method to check if the input array has the supported dimensions
        :param im: input image [batch, channel, X, Y, Z]
        :return: does not return anything; will throw an error if the input image is not of a supported dimension
        """
        _ = self.get_spatial_dimension(im)

    def check_supported_boundary_behavior(self, bc):
        """
        Method to check if the selected boundary condition is supported
        :param bc:
        :return: does not return anything; will throw and error if the boundary condition is not supported
        """
        if bc not in BoundaryBehavior:
            raise ValueError("Unsupported boundary condition: {}".format(bc))

    def xp(self, im, bc=BoundaryBehavior.ZERO):
        """
        Returns values where the x coordinate has been evaluated at +1
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Image with values at an x-index one larger
        """
        self.check_supported_dimensions(im)
        self.check_supported_boundary_behavior(bc)
        rxp = torch.zeros_like(im)
        rxp[:, :, 0:-1, ...] = im[:, :, 1:, ...]

        if bc is BoundaryBehavior.ZERO:
            pass  # already initialized to zero, so nothing to do here
        elif bc is BoundaryBehavior.COPY:
            rxp[:, :, -1, ...] = im[:, :, -1, ...]
        elif bc is BoundaryBehavior.REFLECT:
            rxp[:, :, -1, ...] = im[:, :, -2, ...]
        else:
            raise ValueError("Unknown boundary behavior: {}".format(bc))

        return rxp

    def xm(self, im, bc=BoundaryBehavior.ZERO):
        """
        Returns values where the x coordinate has been evaluated at -1
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Image with values at an x-index one smaller
        """
        self.check_supported_dimensions(im)
        self.check_supported_boundary_behavior(bc)
        rxm = torch.zeros_like(im)
        rxm[:, :, 1:, ...] = im[:, :, 0:-1, ...]

        if bc is BoundaryBehavior.ZERO:
            pass  # already initialized to zero, so nothing to do here
        elif bc is BoundaryBehavior.COPY:
            rxm[:, :, 0, ...] = im[:, :, 0, ...]
        elif bc is BoundaryBehavior.REFLECT:
            rxm[:, :, 0, ...] = im[:, :, 1, ...]
        else:
            raise ValueError("Unknown boundary behavior: {}".format(bc))

        return rxm

    def yp(self, im, bc=BoundaryBehavior.ZERO):
        """
        Returns values where the y coordinate has been evaluated at +1
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Image with values at y-index one larger
        """
        self.check_supported_dimensions(im)
        self.check_supported_boundary_behavior(bc)
        ryp = torch.zeros_like(im)
        ryp[:, :, :, 0:-1, ...] = im[:, :, :, 1:, ...]

        if bc is BoundaryBehavior.ZERO:
            pass  # already initialized to zero, so nothing to do here
        elif bc is BoundaryBehavior.COPY:
            ryp[:, :, :, -1, ...] = im[:, :, :, -1, ...]
        elif bc is BoundaryBehavior.REFLECT:
            ryp[:, :, :, -1, ...] = im[:, :, :, -2, ...]
        else:
            raise ValueError("Unknown boundary behavior: {}".format(bc))

        return ryp

    def ym(self, im, bc=BoundaryBehavior.ZERO):
        """
        Returns values where the y coordinate has been evaluated at -1
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Image with values at y-index one smaller
        """
        self.check_supported_dimensions(im)
        self.check_supported_boundary_behavior(bc)
        rym = torch.zeros_like(im)
        rym[:, :, :, 1:, ...] = im[:, :, :, 0:-1, ...]

        if bc is BoundaryBehavior.ZERO:
            pass  # already initialized to zero, so nothing to do here
        elif bc is BoundaryBehavior.COPY:
            rym[:, :, :, 0, ...] = im[:, :, :, 0, ...]
        elif bc is BoundaryBehavior.REFLECT:
            rym[:, :, :, 0, ...] = im[:, :, :, 1, ...]
        else:
            raise ValueError("Unknown boundary behavior: {}".format(bc))

        return rym

    def zp(self, im, bc=BoundaryBehavior.ZERO):
        """
        Returns values where the z coordinate has been evaluated at +1
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Image with values at z-index one larger
        """
        self.check_supported_dimensions(im)
        self.check_supported_boundary_behavior(bc)
        rzp = torch.zeros_like(im)
        rzp[:, :, :, :, 0:-1] = im[:, :, :, :, 1:]

        if bc is BoundaryBehavior.ZERO:
            pass  # already initialized to zero, so nothing to do here
        elif bc is BoundaryBehavior.COPY:
            rzp[:, :, :, :, -1] = im[:, :, :, :, -1]
        elif bc is BoundaryBehavior.REFLECT:
            rzp[:, :, :, :, -1] = im[:, :, :, :, -2]
        else:
            raise ValueError("Unknown boundary behavior: {}".format(bc))

        return rzp

    def zm(self, im, bc=BoundaryBehavior.ZERO):
        """
        Returns values where the z coordinate has been evaluated at -1
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Image with values at z-index one smaller
        """
        self.check_supported_dimensions(im)
        self.check_supported_boundary_behavior(bc)
        rzm = torch.zeros_like(im)
        rzm[:, :, :, :, 1:] = im[:, :, :, :, 0:-1]

        if bc is BoundaryBehavior.ZERO:
            pass  # already initialized to zero, so nothing to do here
        elif bc is BoundaryBehavior.COPY:
            rzm[:, :, :, :, 0] = im[:, :, :, :, 0]
        elif bc is BoundaryBehavior.REFLECT:
            rzm[:, :, :, :, 0] = im[:, :, :, :, 1]
        else:
            raise ValueError("Unknown boundary behavior: {}".format(bc))

        return rzm

    def dxb(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Backward difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_i-I_{i-1}}{h_x}`

        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in x direction using backward differences
        """

        if bc is BoundaryCondition.DIRICHLET_ZERO:
            res = (im - self.xm(im, bc=BoundaryBehavior.ZERO)) / self.spacing[0]
        elif bc is BoundaryCondition.NEUMANN_ZERO:
            res = (im - self.xm(im, bc=BoundaryBehavior.COPY)) / self.spacing[0]
        else:
            raise ValueError("Unknown boundary condition: {}".format(bc))

        return res

    def dxf(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Forward difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_{i+1}-I_{i}}{h_x}`

        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in x direction using forward differences
        """

        if bc is BoundaryCondition.DIRICHLET_ZERO:
            res = (self.xp(im, bc=BoundaryBehavior.ZERO) - im) / self.spacing[0]
        elif bc is BoundaryCondition.NEUMANN_ZERO:
            res = (self.xp(im, bc=BoundaryBehavior.COPY) - im) / self.spacing[0]
        else:
            raise ValueError("Unknown boundary condition: {}".format(bc))

        return res

    def dxc(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Central difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_{i+1}-I_{i-1}}{2h_x}`

        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in x direction using central differences
        """

        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.xp(im, bc=BoundaryBehavior.COPY)
                - self.xm(im, bc=BoundaryBehavior.COPY)
            ) / (2 * self.spacing[0])
            # now set the results zero on the left and the right
            res[:, :, 0, ...] = 0
            res[:, :, -1, ...] = 0
        else:
            raise ValueError("dxc only supports Neumann boundary conditions.")

        return res

    def ddx(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Second derivative in x direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the second derivative in x direction
        """

        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.xp(im, bc=BoundaryBehavior.COPY)
                - 2 * im
                + self.xm(im, bc=BoundaryBehavior.COPY)
            ) / (self.spacing[0] ** 2)
            # now set the to a [1 -1] for the boundary condition
            res[:, :, 0, ...] = (im[:,:,1,...]-im[:,:,0,...])/(self.spacing[0]**2)
            res[:, :, -1, ...] = (-im[:,:,-1,...] + im[:,:,-2,...])/(self.spacing[0]**2)
        else:
            raise ValueError("ddx (1D) only supports Neumann or Dirichlet boundary conditions.")

        return res

    def dyb(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as dxb, but for the y direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in y direction using backward differences
        """
        if bc is BoundaryCondition.DIRICHLET_ZERO:
            res = (im - self.ym(im, bc=BoundaryBehavior.ZERO)) / self.spacing[1]
        elif bc is BoundaryCondition.NEUMANN_ZERO:
            res = (im - self.ym(im, bc=BoundaryBehavior.COPY)) / self.spacing[1]
        else:
            raise ValueError("Unknown boundary condition: {}".format(bc))

        return res

    def dyf(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as dxf, but for the y direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in y direction using forward differences
        """
        if bc is BoundaryCondition.DIRICHLET_ZERO:
            res = (self.yp(im, bc=BoundaryBehavior.ZERO) - im) / self.spacing[1]
        elif bc is BoundaryCondition.NEUMANN_ZERO:
            res = (self.yp(im, bc=BoundaryBehavior.COPY) - im) / self.spacing[1]
        else:
            raise ValueError("Unknown boundary condition: {}".format(bc))

        return res

    def dyc(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as dxc, but for the y direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in y direction using central differences
        """
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.yp(im, bc=BoundaryBehavior.COPY)
                - self.ym(im, bc=BoundaryBehavior.COPY)
            ) / (2 * self.spacing[1])
            # now set the results zero on the left and the right
            res[:, :, :, 0, ...] = 0
            res[:, :, :, -1, ...] = 0
        else:
            raise ValueError("dyc only supports Neumann boundary conditions.")

        return res

    def ddy(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as ddx, but for the y direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the second derivative in the y direction
        """
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.yp(im, bc=BoundaryBehavior.COPY)
                - 2 * im
                + self.ym(im, bc=BoundaryBehavior.COPY)
            ) / (self.spacing[1] ** 2)
            # now set the to a [1 -1] for the boundary condition
            res[:, :, :, 0, ...] = (im[:, :, :, 1, ...] - im[:, :, :, 0, ...]) / (self.spacing[1] ** 2)
            res[:, :, :, -1, ...] = (-im[:, :, :, -1, ...] + im[:, :, :, -2, ...]) / (self.spacing[1] ** 2)
        else:
            raise ValueError("ddy only supports Neumann boundary conditions.")

        return res

    def dzb(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as dxb, but for the z direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in the z direction using backward differences
        """
        if bc is BoundaryCondition.DIRICHLET_ZERO:
            res = (im - self.zm(im, bc=BoundaryBehavior.ZERO)) / self.spacing[2]
        elif bc is BoundaryCondition.NEUMANN_ZERO:
            res = (im - self.zm(im, bc=BoundaryBehavior.COPY)) / self.spacing[2]
        else:
            raise ValueError("Unknown boundary condition: {}".format(bc))

        return res

    def dzf(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as dxf, but for the z direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in the z direction using forward differences
        """
        if bc is BoundaryCondition.DIRICHLET_ZERO:
            res = (self.zp(im, bc=BoundaryBehavior.ZERO) - im) / self.spacing[2]
        elif bc is BoundaryCondition.NEUMANN_ZERO:
            res = (self.zp(im, bc=BoundaryBehavior.COPY) - im) / self.spacing[2]
        else:
            raise ValueError("Unknown boundary condition: {}".format(bc))

        return res

    def dzc(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as dxc, but for the z direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the first derivative in the z direction using central differences
        """
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.zp(im, bc=BoundaryBehavior.COPY)
                - self.zm(im, bc=BoundaryBehavior.COPY)
            ) / (2 * self.spacing[2])
            # now set the results zero on the left and the right
            res[:, :, :, :, 0] = 0
            res[:, :, :, :, -1] = 0
        else:
            raise ValueError("dzc only supports Neumann boundary conditions.")

        return res

    def ddz(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Same as ddx, but for the z direction
        :param im: Input image [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the second derivative in the z direction
        """
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.zp(im, bc=BoundaryBehavior.COPY)
                - 2 * im
                + self.zm(im, bc=BoundaryBehavior.COPY)
            ) / (self.spacing[2] ** 2)
            # now set the to a [1 -1] for the boundary condition
            res[:, :, :, :, 0] = (im[:, :, :, :, 1] - im[:, :, :, :, 0]) / (self.spacing[2] ** 2)
            res[:, :, :, :, -1] = (-im[:, :, :, :, -1] + im[:, :, :, :, -2]) / (self.spacing[2] ** 2)
        else:
            raise ValueError("ddz only supports Neumann boundary conditions.")

        return res

    def ddxy(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Cross derivative for x and y
        :param im: Input image  [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the cross derivative in xy directions
        """
        # TODO: make this symmetric with respect to the boundary conditions
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.xp(
                    self.yp(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                + self.xm(
                    self.ym(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                - self.xp(
                    self.ym(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                - self.xm(
                    self.yp(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
            ) / (4 * self.spacing[0] * self.spacing[1])
        else:
            raise ValueError("ddxy only supports Neumann boundary conditions.")

        return res

    def ddxz(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Cross derivative for x and z
        :param im: Input image  [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the cross derivative in xy directions
        """
        # TODO: make this symmetric with respect to the boundary conditions
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.xp(
                    self.zp(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                + self.xm(
                    self.zm(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                - self.xp(
                    self.zm(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                - self.xm(
                    self.zp(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
            ) / (4 * self.spacing[0] * self.spacing[2])
        else:
            raise ValueError("ddxz only supports Neumann boundary conditions.")

        return res

    def ddyz(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Cross derivative for y and z
        :param im: Input image  [batch, channel, X, Y, Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the cross derivative in xy directions
        """
        # TODO: make this symmetric with respect to the boundary conditions
        if bc is BoundaryCondition.NEUMANN_ZERO:
            res = (
                self.yp(
                    self.zp(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                + self.ym(
                    self.zm(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                - self.yp(
                    self.zm(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
                - self.ym(
                    self.zp(im, bc=BoundaryBehavior.COPY), bc=BoundaryBehavior.COPY
                )
            ) / (4 * self.spacing[1] * self.spacing[2])
        else:
            raise ValueError("ddz only supports Neumann boundary conditions.")

        return res

    def lap(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Compute the Laplacian of an image
        :param im: Input image [batch, channel, X,Y,Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: Returns the Laplacian
        """
        dim = self.get_spatial_dimension(im)
        if dim == 1:
            return self.ddx(im, bc=bc)
        elif dim == 2:
            return self.ddx(im, bc=bc) + self.ddy(im, bc=bc)
        elif dim == 3:
            return self.ddx(im, bc=bc) + self.ddy(im, bc=bc) + self.ddz(im, bc=bc)
        else:
            raise ValueError(
                "Finite differences are only supported in dimensions 1 to 3"
            )

    def grad_norm_sqr_c(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Computes the gradient norm of an image
        :param im: Input image [batch, channel, X,Y,Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: returns ||grad I||^2
        """
        dim = self.get_spatial_dimension(im)
        if dim == 1:
            return self.dxc(im, bc=bc) ** 2
        elif dim == 2:
            return self.dxc(im, bc=bc) ** 2 + self.dyc(im, bc=bc) ** 2
        elif dim == 3:
            return (
                self.dxc(im, bc=bc) ** 2
                + self.dyc(im, bc=bc) ** 2
                + self.dzc(im, bc=bc) ** 2
            )
        else:
            raise ValueError(
                "Finite differences are only supported in dimensions 1 to 3"
            )

    def grad_norm_sqr_f(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Computes the gradient norm of an image
        :param im: Input image [batch, channel, X,Y,Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: returns ||grad I||^2
        """
        dim = self.get_spatial_dimension(im)
        if dim == 1:
            return self.dxf(im, bc=bc) ** 2
        elif dim == 2:
            return self.dxf(im, bc=bc) ** 2 + self.dyf(im, bc=bc) ** 2
        elif dim == 3:
            return (
                self.dxf(im, bc=bc) ** 2
                + self.dyf(im, bc=bc) ** 2
                + self.dzf(im, bc=bc) ** 2
            )
        else:
            raise ValueError(
                "Finite differences are only supported in dimensions 1 to 3"
            )

    def grad_norm_sqr_b(self, im, bc=BoundaryCondition.NEUMANN_ZERO):
        """
        Computes the gradient norm of an image
        :param im: Input image [batch, channel, X,Y,Z]
        :param bc: controls the boundary behavior of the finite difference (BoundaryBehavior)
        :return: returns ||grad I||^2
        """
        dim = self.get_spatial_dimension(im)
        if dim == 1:
            return self.dxb(im, bc=bc) ** 2
        elif dim == 2:
            return self.dxb(im, bc=bc) ** 2 + self.dyb(im, bc=bc) ** 2
        elif dim == 3:
            return (
                self.dxb(im, bc=bc) ** 2
                + self.dyb(im, bc=bc) ** 2
                + self.dzb(im, bc=bc) ** 2
            )
        else:
            raise ValueError(
                "Finite differences are only supported in dimensions 1 to 3"
            )

