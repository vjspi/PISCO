''' Espirit implementation adapted from https://github.com/mikgroup/espirit-python/blob/master/espirit.py'''
import medutils.mri
import torch
import numpy as np
from tqdm import tqdm
import medutils.visualization as vis
from utils.basic import torch2numpy, numpy2torch
from utils.mri import scale_traj, mriRadialAdjointOp
import matplotlib.pyplot as plt

fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax)
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax)


def espirit_from_radial_wrapper(kdata, traj, im_size, k=6, r=20, t=0.01, c=0.9925, ncalib=None,
                                device="cpu", chunk_size=10, debug=False):
    """
    ESPIRiT Coil Sensitivity Map Estimation Wrapper for Radial MRI Reconstruction

    Parameters:
    - kdata (np.ndarray): Input k-space data. Expected shape is [e, c, sl, pe, fe].
    - traj (np.ndarray): Radial trajectory data. Expected shape is [e, pe, fe, 2].
    - im_size (tuple): Desired output image size. Should be in the format (x, y, z).
    - k (int): ESPIRiT parameter controlling kernel size (default: 6).
    - r (int): ESPIRiT parameter controlling rank of coil sensitivity maps (default: 20).
    - t (float): ESPIRiT parameter controlling threshold for singular values (default: 0.01).
    - c (float): ESPIRiT parameter controlling regularization (default: 0.9925).
    - ncalib (int): Limits the region for sensitivity map estimation. If None, defaults to im_size[0].
    - device (str or torch.device): The device on which to perform computations (default: "cpu").
    - chunk_size (int): Size of chunks for processing k-space slices (default: 10).
    - debug (bool): If True, enables debug mode with visualization (default: False).

    Returns:
    - torch.Tensor: Coil sensitivity maps (CSMs) as torch.Tensor of shape [x, y, z, c].
    - torch.Tensor: Coil images obtained with INUFFT as torch.Tensor of shape [e, c, sl, x, y].

    Note:
    - kspace input is expected as [e, c, sl, pe, fe].
    - traj input is expected as [e, pe, fe, 2].
    - ncalib limits the region for sensitivity map estimation; output of ESPIRiT is of this size and will be interpolated back to im_size.
    - The coil-combined image is returned for visualization purposes.
    """

    ncalib = im_size[0] if ncalib is None else ncalib

    ## Prepare data for torchkbnufft
    nEch, nCoils, nSl, nPE, nFE = kdata.shape
    kdata = torch.from_numpy(kdata).to(dtype=torch.complex64)
    traj = torch.from_numpy(traj).to(dtype=torch.float32)
    # traj = traj * torch.pi     # rescale for tkbn -pi to pi
    traj = traj.permute(0, 3, 1, 2)  # nechos, 2, pe, fe

    kdata, traj = kdata.reshape(*kdata.shape[:-2], -1), traj.reshape(*traj.shape[:-2], -1)

    #### Create cartesian kspace using torchkbnufft
    ## Get image with INUFFT
    from utils.mri import mriRadialAdjointOp
    ech = 0                                                 # only on one echo
    img = torch.zeros(1, nCoils, nSl, *im_size, dtype=torch.complex64, device=device)
    for sl in range(nSl):
        img[...,sl, :, :] = mriRadialAdjointOp(kdata[[ech], :, sl, ...], shape=im_size,
                             traj=traj[[ech], ...], dcf="calc",
                             csm=None, osf=2, chunk_size=10, device=device)
    if debug:
        from utils.vis import angle2color
        vis.imshow(vis.contrastStretching(vis.plot_array(torch2numpy(img[0, :, 0, ...]))), title="NUFFT:Coil images")
        vis.imshow(vis.plot_array(np.angle(torch2numpy(img[0, :, 0, ...]))), title="NUFFT:Coil images - phase")
        vis.imshow(angle2color(vis.plot_array(np.angle(torch2numpy(img[0, :, 0, ...])))), title="NUFFT:Coil images - phase")
        plt.show()

    ## Create cartesian kspace from image with FFT
    axes = (-1, -2)
    kdata_cart = fft_torch(img, ax=axes)
    if debug:
        img_cart = ifft_torch(kdata_cart, ax=axes)
        vis.kshow(vis.plot_array(torch2numpy(kdata_cart[0,:,0, ...])), title="Cartesian k-space (k -> nufft -> fft)")
        vis.imshow(vis.plot_array(vis.contrastStretching(np.abs(torch2numpy(img_cart[0, :, 0, ...])))), title="Img from Cartesian k-space (k -> nufft -> fft -> ifft)")
        vis.imshow(vis.plot_array(np.angle(torch2numpy(img_cart[0, :, 0, ...]))), title="Img from Cartesian k-space (k -> nufft -> fft -> ifft)")
        plt.show()

    ### Optional: process kdata_cart to remove extrapolated k-space values (e.g. outside of the acquired circle of radial trajectory)
    # center = kdata_cart.shape[-1]//2
    # radius = kdata_cart.shape[-1] - center
    # x, y = torch.meshgrid(torch.arange(-radius, radius), torch.arange(-radius, radius))
    # mask = torch.sqrt(x ** 2 + y ** 2) <= radius
    # mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1, kdata.shape[1], 1, -1, -1)
    # kdata_cart[~mask] = 0j


    ## Estimate sensmaps from cartesian kspace data
    kdata_cart = medutils.visualization.center_crop(kdata_cart, shape=[ncalib, ncalib])    # Crop calibration region
    kdata_cart = kdata_cart[ech, ...].permute(2,3,1,0)              # x,y,z,c # reshape for espirit
    smaps = espirit_torch_slicewise_wrapper(kdata_cart, device=device, chunk_size=chunk_size,
                                            k=k, r=r, t=t, c=c, espirit_numpy=True)

    ## Interpolate sensmaps if calibrated on smaller kspace
    if ncalib != im_size[0]:
        from scipy import ndimage
        ## need to interpolate the coil map
        factor = im_size[0] / ncalib
        smaps = torch2numpy(smaps)
        smaps_inter = np.zeros((*im_size, *smaps.shape[2:]), dtype=smaps.dtype)
        for c in range(smaps.shape[-1]):
            for z in range(smaps.shape[-2]):
                smaps_inter[:,:,z,c] = ndimage.zoom(smaps[:,:,z,c], zoom=factor)
        smaps = numpy2torch(smaps_inter.copy(), device=device)

    if debug:
        from utils.mri import coilcombine
        from utils.vis import angle2color
        vis.imshow(vis.plot_array(torch2numpy(smaps[...,0,:]).transpose(2,0,1)), title="NUFFT:coil sensitivity maps - Mag")
        vis.imshow(vis.plot_array(np.angle(torch2numpy(smaps[...,0,:]).transpose(2,0,1))), title="NUFFT:coil sensitivity maps - Phase")
        plt.show()

        img_cc = coilcombine(img[0,:, 0,...].permute(1,2,0), csm=smaps[...,0,:], coil_dim=-1)
        vis.imshow(vis.contrastStretching(torch2numpy(img_cc)),title="NUFFT:Coil combined - mag")
        img_cc_angle = angle2color(np.angle(torch2numpy(img_cc)), cmap="gray", vmin=-np.pi, vmax=np.pi)
        vis.imshow(img_cc_angle, title="NUFFT:Coil combined - angle")
        plt.show()

    return smaps, img

def espirit_torch_slicewise_wrapper(kdata, device, k=6, r=24, t=0.01, c=0.9925, chunk_size = 10, espirit_numpy=True):
    """
    ESPIRiT Coil Sensitivity Map Estimation Wrapper for PyTorch, Slice-Wise

    Parameters:
    - kdata (torch.Tensor): Input k-space data. Expected shape is [x, y, z, c].
    - device (torch.device): The device on which to perform computations.
    - k (int): ESPIRiT parameter controlling kernel size (default: 6).
    - r (int): ESPIRiT parameter controlling rank of coil sensitivity maps (default: 24).
    - t (float): ESPIRiT parameter controlling threshold for singular values (default: 0.01).
    - c (float): ESPIRiT parameter controlling regularization (default: 0.9925).
    - chunk_size (int): Size of chunks for processing k-space slices (default: 10).
    - espirit_numpy (bool): Determines whether ESPIRiT coil estimation is conducted with NumPy (recommended). If False, conducted with PyTorch.

    Returns:
    - torch.Tensor: Coil sensitivity maps (CSMs) as torch.Tensor of shape [x, y, z, c].

    Note:
    - kspace input is expected as [x, y, z, c] and kspace is expected to be centered.
    - If espirit_numpy is set to True, ESPIRiT coil estimation is performed using NumPy (recommended).
    - If espirit_numpy is set to False, ESPIRiT coil estimation is performed using PyTorch.
    - Care should be taken when using PyTorch for coil estimation as torch.linalg.svd may not produce reasonable phase for sensitivity maps.
    """
    ### Preprocess data
    if espirit_numpy:
        kdata=torch2numpy(kdata)
        smaps_espirit = np.zeros_like(kdata)  # [c,x,y,z]
    else:
        kdata = kdata.to(device)
        smaps_espirit = torch.zeros_like(kdata)  # [c,x,y,z]

    vis.kshow(vis.plot_array(torch2numpy(kdata[..., 0, :]).transpose(2, 0, 1)), title="Espirit_kspace_input")
    def generate_smaps_slicewise(kdata, nSlices):
        for sl in range(nSlices):
            smaps = espirit_func(kdata[...,[sl],:])
            yield smaps

    nSl = kdata.shape[-2]
    for start in tqdm(range(0, nSl, chunk_size), desc='Processing kspace slices with espirit'.format(chunk_size), leave=False):
        end = start + min(chunk_size, nSl-start)
        if espirit_numpy:
            espirit_func = lambda kdata: espirit(kdata, k, r, t, c)[..., :, 0]
            smaps_espirit[..., start:end, :] = np.concatenate(list(generate_smaps_slicewise(kdata[...,start:end, :],
                                                          nSlices=end-start)), axis = -2)
        else:
            espirit_func = lambda kdata: espirit_torch(kdata, k, r, t, c)[..., :, 0]
            smaps_espirit[..., start:end, :] = torch.cat(list(generate_smaps_slicewise(kdata[...,start:end, :],
                                                          nSlices=end-start)), dim = -2)

    return numpy2torch(smaps_espirit, device=device) if espirit_numpy else smaps_espirit


def espirit(X, k, r, t, c):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """

    sx = np.shape(X)[0]
    sy = np.shape(X)[1]
    sz = np.shape(X)[2]
    nc = np.shape(X)[3]

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
    szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)

    # Extract calibration region.
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].astype(np.complex64)

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = np.zeros([(r-k+1)**p, k**p * nc]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
      for ydx in range(max(1, C.shape[1] - k + 1)):
        for zdx in range(max(1, C.shape[2] - k + 1)):
          # numpy handles when the indices are too big
          block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].astype(np.complex64)
          A[idx, :] = block.flatten()
          idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    n = np.sum(S >= t * S[0])
    V = V[:, 0:n]

    kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
    kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
    kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
            kerimgs[:,:,:,jdx,idx] = fft(ker, axes) * np.sqrt(sx * sy * sz)/np.sqrt(k**p)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):

                Gq = kerimgs[idx,jdx,kdx,:,:]

                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if (s[ldx]**2 > c):
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]

    return maps

fft_torch  = lambda x, ax : torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x,  dim= ax), dim= ax, norm='ortho'),  dim= ax)
ifft_torch = lambda X, ax : torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X,  dim= ax),  dim= ax, norm='ortho'),  dim= ax)

def espirit_torch(X, k, r, t, c):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """
    device = X.device  # Get the device of the input tensor

    # Get the dimensions of X
    sx, sy, sz, nc = X.shape

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
    szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)

    # Extract calibration region.
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].to(device=device, dtype=torch.complex64)

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = torch.zeros([(r-k+1)**p, k**p * nc]).to(device=device, dtype=torch.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
      for ydx in range(max(1, C.shape[1] - k + 1)):
        for zdx in range(max(1, C.shape[2] - k + 1)):
          # numpy handles when the indices are too big
          block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].to(device=device, dtype=torch.complex64)
          A[idx, :] = block.flatten()
          idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = torch.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    n = torch.sum(S >= t * S[0])
    V = V[:, 0:n]

    kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
    kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
    kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = torch.zeros((*X.shape, n), device=device, dtype=torch.complex64)
    kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = torch.reshape(V[:, idx], kerdims)

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = torch.zeros((*X.shape, n), device=device, dtype=torch.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = torch.flip(kernels[:, :, :, jdx, idx].conj(), dims=(0, 1, 2))
            kerimgs[:,:,:,jdx,idx] = fft_torch(ker, axes) * np.sqrt(sx * sy * sz)/np.sqrt(k**p)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = torch.zeros((*X.shape, n), device=device, dtype=torch.complex64)
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):

                Gq = kerimgs[idx,jdx,kdx,:,:]

                u, s, vh = torch.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if (s[ldx]**2 > c):
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]

    return maps


# def interpolate_smaps(smaps, im_size):
#     '''
#     smaps: exptected shape
#     '''
#     smaps = torch2numpy(smaps)
#     dim = len(im_size)
#     if smaps.shape[0:dim] != im_size:  # Check if the sensitivities map size matches the image size
#         # Create a new array to store the interpolated sensitivities maps
#         interpolated_smaps = np.zeros((*im_size, *smaps.shape[dim:]), dtype=smaps.dtype)
#
#         if dim == 2:
#             for z in range(smaps.shape[2]):
#                 # Iterate over the channels (assuming the last dimension represents channels)
#                 for c in range(smaps.shape[-1]):
#                     # Create a 2D interpolation function for each channel
#                     f = interpolate.interp2d(np.arange(smaps.shape[1]), np.arange(smaps.shape[0]), smaps[:, :, z, c],
#                                              kind='linear')
#
#                     # Generate new grid coordinates for the interpolation
#                     x_new = np.linspace(0, smaps.shape[0] - 1, im_size[0])
#                     y_new = np.linspace(0, smaps.shape[1] - 1, im_size[1])
#
#                     # Perform the interpolation
#                     interpolated_smaps[:, :, z, c] = f(y_new, x_new)
#         elif dim == 3:
#             for c in range(smaps.shape[-1]):
#                 # Create a 2D interpolation function for each channel
#                 f = interpolate.interp3d(np.arange(smaps.shape[2]), np.arange(smaps.shape[1]),
#                                          np.arange(smaps.shape[0]), smaps[:, :, :, c],
#                                          kind='linear')
#
#                 # Generate new grid coordinates for the interpolation
#                 x_new = np.linspace(0, smaps.shape[0] - 1, im_size[0])
#                 y_new = np.linspace(0, smaps.shape[1] - 1, im_size[1])
#                 z_new = np.linspace(0, smaps.shape[2] - 1, im_size[2])
#
#                 # Perform the interpolation
#                 interpolated_smaps[:, :, :, c] = f(x_new, y_new, z_new)
#
#         return interpolated_smaps
#     else:
#         return smaps
