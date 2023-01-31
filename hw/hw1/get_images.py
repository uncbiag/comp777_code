import torch

def get_images_mean_sigmas_proportion( mean1, mean2, std1, std2, prop1 ):
    I = torch.zeros(200, 200)
    I_noise = torch.zeros(200, 200)

    to1 = round( 200*prop1 )

    I[0:to1,:] = mean1
    I[to1:-1,:] = mean2

    I_noise[0:to1,:] = I[0:to1,:] + std1*torch.randn(to1,200)
    I_noise[to1:,:] = I[to1:,:] + std2*torch.randn(200-to1,200)

    I_seg = I > mean1

    return I_noise, I, I_seg

def get_images_mean_sigmas( mean1, mean2, std1, std2):
    I = torch.zeros(200, 200)
    I_noise = torch.zeros(200, 200)

    I[0:100,:] = mean1
    I[100:-1,:] = mean2

    I_noise[0:100,:] = I[0:100,:] + std1*torch.randn(100,200)
    I_noise[100:,:] = I[100:,:] + std2*torch.randn(100,200)

    I_seg = I>mean1

    return I_noise, I, I_seg

def get_images( nr ):
    # nr: specifies the desired scenario (1-4)
    # returns
    # I_noise, I, I_seg, means, stds, prop
    # I_noise: noisy image
    # ISeg: ground - truth segmentation for validation
    # I: image without noise
    # means: means for foreground and background partition (do not use for homework)
    # stds: standard deviations for foreground and background partitions (do not use for homework)
    # prop: percentage of the overall area which is background

    prop = 0.5

    if nr==1:
        I_noise, I, I_seg = get_images_mean_sigmas( mean1=100, mean2=200, std1=25, std2=25 )
        means = [100, 200]
        stds = [25, 25]
    elif nr==2:
        I_noise, I, I_seg = get_images_mean_sigmas( mean1=100, mean2=200, std1=5, std2=50 )
        means = [100, 200]
        stds = [5, 50]
    elif nr==3:
        I_noise, I, I_seg = get_images_mean_sigmas_proportion( mean1=100, mean2=200, std1=25, std2=25, prop1=0.05 )
        means = [100, 200]
        stds = [25, 25]
        prop = 0.05
    elif nr==4:
        I_noise, I, I_seg = get_images_mean_sigmas_proportion( mean1=100, mean2=200, std1=5, std2=50, prop1=0.05 )
        means = [100, 200]
        stds = [5, 50]
        prop = 0.05
    else:
        raise ValueError('Unknown scenario')

    return I_noise, I, I_seg, means, stds, prop

def plot_scenario( I_noise, I, I_seg):

    import holoviews as hv
    hv.extension('bokeh')
    from bokeh.plotting import show

    im_I = hv.Image( I.numpy()).opts(colorbar=True, xaxis=None, yaxis=None)
    im_I_noise = hv.Image( I_noise.numpy()).opts(colorbar=True, xaxis=None, yaxis=None)
    im_I_seg = hv.Image( I_seg.numpy()).opts(colorbar=False, xaxis=None, yaxis=None)

    show(hv.render( im_I + im_I_seg + im_I_noise ))

def test_me():

    I_noise1, I1, I_seg1, _, _, _ = get_images(nr=1)
    plot_scenario( I_noise1, I1, I_seg1)

    I_noise2, I2, I_seg2, _, _, _ = get_images(nr=2)
    plot_scenario(I_noise2, I2, I_seg2)

    I_noise3, I3, I_seg3, _, _, _ = get_images(nr=3)
    plot_scenario(I_noise3, I3, I_seg3)

    I_noise4, I4, I_seg4, _, _, _ = get_images(nr=4)
    plot_scenario(I_noise4, I4, I_seg4)

