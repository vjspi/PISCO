import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from medutils.visualization import center_crop, contrastStretching, plot_array, imsave
from utils.mri import coilcombine, ifft2c_mri, clip_to_percentiles
from utils.basic import numpy2torch, torch2numpy
from PIL import Image, ImageDraw, ImageFont
import medutils
import wandb


def kshow_nolog(kspace, title="", offset=1e-4, figsize=None):
    """ Show k-space
    :param kspace: input k-space (np.array)
    :param title: plot title
    :param offset: offset for log scale
    """
    img = np.abs(kspace)
    img /= np.max(img)
    # img = np.log(img + offset)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()


def k2img_multiecho(k, csm=None, norm_factor=1, scale = True, multi_coil=False):
    """
    Convert k-space data to image space and generate visual representations.

    Parameters:
        k (torch.Tensor or numpy.ndarray): k-space data on a Cartesian grid. The dimensions can be either 4 or 5.
        csm (torch.Tensor or None, optional): Coil sensitivity maps. If provided, coil combination will be performed using the maps.
        norm_factor (float, optional): Normalization factor for image scaling (default is 1.5).
        scale (bool, optional): Whether to scale the output images to the range [0, 255] (default is True).

    Returns:
        dict: A dictionary containing different visual representations of the data:
            - 'k_mag': Magnitude of the k-space data.
            - 'combined_mag': Combined magnitude image after coil combination.
            - 'combined_phase': Combined phase image after coil combination.
            - 'combined_img': Combined complex image after coil combination.
    """

    assert k.ndim >= 5
    n_coils = k.shape[2]
    n_echoes = k.shape[1]
    n_dyn = k.shape[0]

    # Fourier transform
    coil_img = torch.empty_like(k)
    for i in range(n_coils):
        coil_img[:, :, i, ...] = ifft2c_mri(k[:, :, i, ...])
    # Combine coils
    if csm is not None:
        combined_img = coilcombine(coil_img, csm=numpy2torch(csm, coil_img.device), coil_dim=2)
    else:
        AssertionError("no coil maps given")
        # combined_img = coilcombine(coil_img, coil_dim=2, mode='rss')
    # Create mag, phase and k space images
    combined_phase = torch.angle(combined_img).detach().cpu().numpy()
    combined_mag = combined_img.abs().detach().cpu().numpy()
    coil_mag = coil_img.abs().detach().cpu().numpy()
    k_mag = k[:, :, ...].abs().detach().cpu().numpy()  # nt, nx, ny
    k_mag = np.log(np.abs(k_mag) + 1e-4) # Log to k_mag to visualize high-freq components

    # find max and min over all echoes and coils
    k_min = np.min(k_mag)
    k_max = np.max(k_mag)
    combined_mag_max = combined_mag.max() / norm_factor
    coil_mag_max = coil_mag.max() / norm_factor

    # Prepare for visualization
    # Single coil
    if scale:
        max_int = 255
        combined_phase_temp = []
        for ech in range(n_echoes):
            k_mag[:,ech,...] = (k_mag[:,ech,...] - k_min) * (max_int) / (k_max - k_min)
            combined_mag[:,ech,...] = (combined_mag[:,ech,...] / combined_mag_max * max_int)  # .astype(np.uint8)
            combined_phase_temp.append(angle2color(combined_phase[:,ech,...] , cmap='gray', vmin=-np.pi, vmax=np.pi))
            coil_mag[:,ech,...] =  coil_mag[:,ech,...] / coil_mag_max * max_int
        combined_phase = np.stack(combined_phase_temp, axis=1)
        # Convert to int
        k_mag = np.clip(k_mag, 0, 255).astype(np.uint8)
        combined_mag = np.clip(combined_mag, 0, 255).astype(np.uint8)
        combined_phase = np.clip(combined_phase, 0, 255).astype(np.uint8)
        coil_mag = np.clip(coil_mag, 0, 255).astype(np.uint8)
        combined_img = combined_img.detach().cpu().numpy()

    vis_dic = {
        'k_mag_coil0': k_mag[:,:,[0],...],
        'combined_mag': combined_mag,
        'combined_phase': combined_phase,
        'combined_img': combined_img
    }

    # Multi_coil
    if multi_coil:
        k_mag_multicoil, coil_mag_multicoil = [], []
        for ech in range(n_echoes):
            k_mag_coils_ech = np.stack(
                [medutils.visualization.plot_array(k_mag[dyn, ech, ...]) for dyn in range(n_dyn)])
            k_mag_multicoil.append(k_mag_coils_ech)
            coil_mag_coils_ech = np.stack(
                [medutils.visualization.plot_array(coil_mag[dyn, ech, ...]) for dyn in range(n_dyn)])
            coil_mag_multicoil.append(coil_mag_coils_ech)
        k_mag_multicoil = np.stack(k_mag_multicoil, axis=1)
        coil_mag_multicoil = np.stack(coil_mag_multicoil, axis=1)
        vis_dic['multicoil_k_mag'] = k_mag_multicoil[:,:,None,...] # add channel dim for video
        vis_dic['multicoil_mag'] = coil_mag_multicoil[:,:,None,...]

    return vis_dic


def k2img(k, csm=None, norm_factor=1.5, scale = True, k_min = None, k_max = None):
    """
    Convert k-space data to image space and generate visual representations.

    Parameters:
        k (torch.Tensor or numpy.ndarray): k-space data on a Cartesian grid. The dimensions can be either 4 or 5.
        csm (torch.Tensor or None, optional): Coil sensitivity maps. If provided, coil combination will be performed using the maps.
        norm_factor (float, optional): Normalization factor for image scaling (default is 1.5).
        scale (bool, optional): Whether to scale the output images to the range [0, 255] (default is True).

    Returns:
        dict: A dictionary containing different visual representations of the data:
            - 'k_mag': Magnitude of the k-space data.
            - 'combined_mag': Combined magnitude image after coil combination.
            - 'combined_phase': Combined phase image after coil combination.
            - 'combined_img': Combined complex image after coil combination.
    """

    if k.ndim == 4:
        coil_img = ifft2c_mri(k)

    elif k.ndim == 5:
        coil_img = torch.empty_like(k)
        for i in range(k.shape[2]):
            coil_img[:, :, i, ...] = ifft2c_mri(k[:, :, i, ...])
        # combined_img_motion = coil_img_motion.abs()

    k_mag = k[:, 0, ...].abs().unsqueeze(1).detach().cpu().numpy()  # nt, nx, ny
    if csm is not None:
        im_shape = csm.shape[-2:]  # (nx, ny)
        combined_img = coilcombine(coil_img, csm, im_size=im_shape, coil_dim=1)
    else:
        combined_img = coilcombine(coil_img, coil_dim=1, mode='rss')
    combined_phase = torch.angle(combined_img).detach().cpu().numpy()
    combined_mag = combined_img.abs().detach().cpu().numpy()
    # k_mag = np.log(np.abs(k_mag) + 1e-4)

    if scale:
        max_int = 255

        # Log to k_mag to visualize high-freq components
        k_mag = np.log(np.abs(k_mag) + 1e-4)
        k_min = np.min(k_mag) if k_min is None else np.log(k_min + 1e-4)
        k_max = np.max(k_mag) if k_max is None else np.log(k_max + 1e-4)
        # scale k_mag
        k_mag = (k_mag - k_min) * (max_int) / (k_max - k_min)
        k_mag = np.minimum(max_int, np.maximum(0.0, k_mag))
        k_mag = k_mag.astype(np.uint8)
        # scale i_mag/i_phase
        combined_mag_max = combined_mag.max() / norm_factor
        combined_mag = (combined_mag / combined_mag_max * 255)  # .astype(np.uint8)
        combined_phase = angle2color(combined_phase, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        # Convert to int
        k_mag = np.clip(k_mag, 0, 255).astype(np.uint8)
        combined_mag = np.clip(combined_mag, 0, 255).astype(np.uint8)
        combined_phase = np.clip(combined_phase, 0, 255).astype(np.uint8)

    combined_img = combined_img.detach().cpu().numpy()
    vis_dic = {
        'k_mag': k_mag,
        'combined_mag': combined_mag,
        'combined_phase': combined_phase,
        'combined_img': combined_img
    }
    return vis_dic



def alpha2img(alpha, csm=None):
    """
    Convert phase data to images suitable for visualization.

    Parameters:
        alpha (torch.Tensor): Phase data (alpha).
        csm (torch.Tensor or None, optional): Coil sensitivity maps. If provided, coil combination will be performed using the maps.

    Returns:
        dict: A dictionary containing visual representations of the phase data:
            - 'alpha': Phase data converted to an image in [0, 255] range.
            - 'alpha_color': Phase data converted to a color image.
    """

    alpha_img = alpha.detach().cpu().numpy()

    alpha_color = angle2color(alpha_img)  # vmin=-1, vmax=1)

    max_int = 255
    alpha_img = (alpha_img - alpha_img.min()) * max_int / (alpha_img.max() - alpha_img.min())
    alpha_img = np.minimum(max_int, np.maximum(0.0, alpha_img))
    alpha_img = alpha_img.astype(np.uint8)

    alpha_vis = {
        'alpha': alpha_img,
        'alpha_color': alpha_color}
    return alpha_vis

def angle2color(value_arr, cmap='viridis', vmin=None, vmax=None):
    """
    Convert a value to a color using a colormap.

    Parameters:
        value: The value to convert.
        cmap: The colormap to use.

    Returns:
        The color corresponding to the value.
    """
    if vmin is None:
        vmin = value_arr.min()
    if vmax is None:
        vmax = value_arr.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    try:
        value_arr = value_arr.squeeze(0)
    except:
        value_arr = value_arr.squeeze()
    if len(value_arr.shape) == 4:
        color_arr = np.zeros((*value_arr.shape, 4))
        for i in range(value_arr.shape[0]):
            for j in range(value_arr.shape[1]):
                color_arr[i,j,...] = mapper.to_rgba(value_arr[i,j,...], bytes=True)
        color_arr = color_arr.transpose(0, 1, 4, 2, 3)
    elif len(value_arr.shape) == 3:
        color_arr = np.zeros((*value_arr.shape, 4))
        for i in range(value_arr.shape[0]):
            color_arr[i] = mapper.to_rgba(value_arr[i], bytes=True)
        color_arr = color_arr.transpose(0, 3, 1, 2)
    elif len(value_arr.shape) == 2:
        color_arr = mapper.to_rgba(value_arr, bytes=True)
    return color_arr


def plot_trajectory_spokes(traj):
    ## expected format [nDyn, nEch, nSpokes, nFe, 2]

    traj_xy = traj
    plot_trajectory(traj_xy)

def plot_trajectory(traj, plot_lim=1, one_plot=False):
    '''
    Function to plot trajectory
    traj: expected format [nDyn, nEch, kx, ky, 2]
    plot_lim: maximum value of x and y axis
    one_plot: if echo dim [dim2]should be plotted in multiple columns or in one subplot
    '''
    ## expected format [nDyn, nEch, kx, ky, 2]

    if isinstance(traj, torch.Tensor):
        traj = traj.cpu().numpy()

    if one_plot == True:
        fig, ax = plt.subplots(nrows=traj.shape[0], ncols=1)
    else:
        fig, ax = plt.subplots(nrows=traj.shape[0], ncols=traj.shape[1])

    if traj.shape[0] == 1 and (traj.shape[1] == 1 or one_plot == True):
        axs = [ax]
    else:
        axs = np.ndarray.flatten(ax)

    if one_plot:
        for d in range(traj.shape[0]):
            for e in range(traj.shape[1]):
                color = plt.cm.jet(e / traj.shape[1])  # Different color for each trajectory in second dimension
                axs[d].plot(traj[d, e, :, :, 1].transpose(), traj[d, e, :, :, 0].transpose(), color=color, alpha=0.5)

            axs[d].axis('equal')
            axs[d].set_xlim([-plot_lim, plot_lim])
            axs[d].set_ylim([-plot_lim, plot_lim])
    else:
        for d in range(traj.shape[0]):
            for e in range(traj.shape[1]):
                idx = d * traj.shape[1] + e
                axs[idx].plot(traj[d,e,:,:,1].transpose(), traj[d,e,:,:,0].transpose(), color="b", alpha=0.1)
                axs[idx].axis('equal')
                axs[idx].set_xlim([-plot_lim, plot_lim])
                axs[idx].set_ylim([-plot_lim, plot_lim])

    plt.show()

def save_gif(img, str=None, filename="debug", numbers_array =None,
             intensity_factor=1, total_duration=100, loop=0, scale_to_255=False):
    r"""
    Save tensor or ndarray as gif.
    Args:
        img: tensor or ndarray, (nt, nx, ny)
        filename: string
        intensity_factor: float, intensity factor for normalizing the data
        duration: int, milliseconds between frames
        loop: int, number of loops, 0 means infinite
    """
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    if len(img.shape) != 3:
        img = img.squeeze(1)
    assert len(img.shape) == 3

    # Normalize image intensity
    if scale_to_255:
        img = (img - img.min()) / (img.max() - img.min())
        img = (np.abs(img) * 255.).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    # img_mag_max = img_mag.max() / intensity_factor
    # # img_mag = (img_mag / img_mag_max * 255)  # .astype(np.uint8)
    # img = (np.abs(img) / img_mag_max * 255).astype(np.uint8)
    # img_mag = np.clip(img_mag, 0, 255)
    # img_mag = np.clip(img_mag, 0, 255).astype(np.uint8)

    # img = (np.abs(img) / np.abs(img).max() * 255 * intensity_factor).astype(np.uint8)
    # img = np.abs(img)/np.abs(img).max()*255#*2
    # img = [Image.fromarray(np.clip(img, 0, 255))]

    if numbers_array is not None:
        assert len(numbers_array) == img.shape[0], "Length of numbers_array must match the number of frames in img"
    # Calculate frame duration
    frame_duration = total_duration // len(img)
    processed_images = []

    # Convert to PIL images and add numbers
    for i in range(len(img)):
        frame = Image.fromarray(img[i])
        if numbers_array is not None:
            frame = add_text_to_image(frame, "t={}s".format(numbers_array[i]), position="bottom-left")
        if str is not None:
            frame = add_text_to_image(frame, str, position="bottom-right")
        processed_images.append(frame)

    # Save as gif
    processed_images[0].save(filename, save_all=True, append_images=processed_images[1:], duration=frame_duration, loop=loop)


def save_pngs_from_npy_path(path_lists, filename, row_names=None, col_names=None, t=14):

    num_rows = len(path_lists)
    num_columns = max(len(sublist) for sublist in path_lists)
    dummy_image = np.load(path_lists[0][0])["array_clipped"]
    image_size = dummy_image.shape[-2:]

    assert len(path_lists) > 0

    processed_images = []
    tot_frame_width = num_columns * image_size[0]    #shapesum(img.shape[1] for img in img_lists[0])
    tot_frame_height = num_rows * image_size[1]       #max(img.shape[2] for img in img_list)

    # Initialize the final image
    final_image = Image.new("RGB", (tot_frame_width, tot_frame_height), (0, 0, 0))
    final_image_phase = Image.new("RGB", (tot_frame_width, tot_frame_height), (0, 0, 0))

    # Load and merge PNGs into one image
    frame_height = 0
    for r, row_paths in enumerate(path_lists):
        row_images = []
        frame_width = 0
        for c, png_path in enumerate(row_paths):
            array = np.load(png_path)["array_clipped"][t,0,...] * 255.
            png_image = Image.fromarray(array.astype('uint8'))
            # if png_image is None:
            #     png_image = Image.fromarray(np.zeros_like(dummy_image))
            # row_images.append(png_image)

            # final_image = Image.new("RGB", (final_image.width, final_image.height + row_images[0].height))
            if c == 0 and row_names is not None:
                add_text_to_image(png_image, row_names[r], position="top-left")
            if r == 0 and col_names is not None:
                add_text_to_image(png_image, col_names[c], position="top")
            # final_image.paste(png_image, (0, 0))
            final_image.paste(png_image, (frame_width, frame_height))

            frame_width += image_size[0]
        frame_height += image_size[1]
    # Save the merged image
    final_image.save(filename)

def save_pngs_from_path(path_lists, filename, row_names=None, col_names=None, total_duration=5000, loop=0):

    num_rows = len(path_lists)
    num_columns = max(len(sublist) for sublist in path_lists)
    dummy_image = Image.open(path_lists[0][0])
    image_size = dummy_image.size

    assert len(path_lists) > 0

    processed_images = []
    tot_frame_width = num_columns * image_size[0]    #shapesum(img.shape[1] for img in img_lists[0])
    tot_frame_height = num_rows * image_size[1]       #max(img.shape[2] for img in img_list)

    # Initialize the final image
    final_image = Image.new("RGB", (tot_frame_width, tot_frame_height), (0, 0, 0))

    # Load and merge PNGs into one image
    frame_height = 0
    for r, row_paths in enumerate(path_lists):
        row_images = []
        frame_width = 0
        for c, png_path in enumerate(row_paths):
            png_image = Image.open(png_path)
            # if png_image is None:
            #     png_image = Image.fromarray(np.zeros_like(dummy_image))
            # row_images.append(png_image)

            # final_image = Image.new("RGB", (final_image.width, final_image.height + row_images[0].height))
            if c == 0 and row_names is not None:
                add_text_to_image(png_image, row_names[r], position="top-left")
            if r == 0 and col_names is not None:
                add_text_to_image(png_image, col_names[c], position="top")
            # final_image.paste(png_image, (0, 0))
            final_image.paste(png_image, (frame_width, frame_height))

            frame_width += image_size[0]
        frame_height += image_size[1]
    # Save the merged image
    final_image.save(filename)


def save_gifs_from_path(path_lists, filename, row_names=None, col_names=None, total_duration=5000, loop=0):

    from PIL import ImageSequence
    num_rows = len(path_lists)
    num_columns = max(len(sublist) for sublist in path_lists)
    # num_frames = img_lists[0][0].shape[0] if img_lists and img_lists[0] else 0

    images_lists = [] # will be list of lists, inner list = colums, outer list = row
    for sublist in path_lists:
        image_row_list = []

        for gif_path in sublist:
            gif_frames = []
            gif = Image.open(gif_path)
            gif_frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(gif)]  # Extract frames
            gif_array = [np.array(frame) for frame in gif_frames]
            gif_array_gray = [(frame_array if len(frame_array.shape)==2 else rgba_to_grayscale(frame_array))
                              for frame_array in gif_array]
            image_row_list.append(np.stack(gif_array_gray, axis=0))

        images_lists.append(image_row_list)

    save_gifs(images_lists, str_lists=None, row_name=row_names, col_name=col_names, numbers_array=None,
              filename=filename, total_duration=total_duration, loop=loop)


def save_gifs(img_lists, str_lists=None, row_name=None, col_name=None,
              filename="debug", numbers_array=None, intensity_factor=1, total_duration=100, loop=0):
    """
     Save multiple lists of image arrays as a gif with columns and rows.
     Args:
         img_lists: List of lists of image arrays.
         str_lists: List of lists of strings for titles (optional).
         filename: String, the output filename.
         numbers_arrays: List of lists of numbers (optional).
         intensity_factor: Float, intensity factor for normalizing the data.
         total_duration: Int, total duration of the gif in milliseconds.
         loop: Int, number of loops, 0 means infinite.
     """
    num_rows = len(img_lists)
    num_columns = max(len(sublist) for sublist in img_lists)
    num_frames = img_lists[0][0].shape[0] if img_lists and img_lists[0] else 0

    assert len(img_lists) > 0

    processed_images = []
    tot_frame_width = num_columns * img_lists[0][0].shape[1]    #shapesum(img.shape[1] for img in img_lists[0])
    tot_frame_height = num_rows * img_lists[0][0].shape[2]      #max(img.shape[2] for img in img_list)
    img_height = img_lists[0][0].shape[2]

    # if str_lists is None:
    #     str_lists = [[None for subentry in sublist] for sublist in (img_lists)]
    # if row_name is None:
    #     str_lists = [None for sublist in (img_lists)]

    for i in range(num_frames):
        frame = Image.new("RGB", (tot_frame_width, tot_frame_height), (0, 0, 0))
        frame_width, frame_height = 0,0

        for row_index, img_row_list in enumerate(img_lists):
            for col_index, img in enumerate(img_row_list):
                title_string = str_lists[row_index][col_index] if str_lists is not None else None
                row_string = row_name[row_index] if row_name is not None else None
                col_string = col_name[col_index] if col_name is not None else None

                if type(img) == torch.Tensor:
                    img = img.cpu().numpy()
                if len(img.shape) != 3:
                    img = img.squeeze(1)
                assert len(img.shape) == 3

                # Normalize image intensity
                # img_mag = np.abs(img[i, ...])
                # img_mag_max = img_mag.max() / intensity_factor
                # img_mag = (img_mag / img_mag_max * 255).astype(np.uint8)

                img_frame = Image.fromarray(img[i,...])
                if numbers_array is not None and i < len(numbers_array):
                    assert len(numbers_array) == img.shape[0]
                    img_frame = add_text_to_image(img_frame, "t={}s".format(numbers_array[i]), position="bottom-left")

                if title_string is not None:
                    img_frame = add_text_to_image(img_frame, title_string, position="bottom-right")

                if row_string is not None and col_index == 0:
                    img_frame = add_text_to_image(img_frame, row_string, position="top-left")

                if col_string is not None and row_index == 0:
                    img_frame = add_text_to_image(img_frame, col_string, position="top")

                frame.paste(img_frame, (frame_width, frame_height))
                frame_width += img_frame.width

            frame_height += img_height  # set to next row
            frame_width = 0                   # reset framewidth

        processed_images.append(frame)

        # Save as gif
    if len(processed_images) > 0:
        frame_duration = total_duration // len(processed_images)
        processed_images[0].save(filename, save_all=True, append_images=processed_images[1:],
                                 duration=frame_duration, loop=loop)

def add_text_to_image(image, text, position="top-left"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font)

    if "top" in position:
        y = 5
    elif "bottom" in position:
        y = image.height - text_height - 5
    else:
        y = (image.height - text_height) // 2

    if "left" in position:
        x = 5
    elif "right" in position:
        x = image.width - text_width - 5
    else:
        x = (image.width - text_width) // 2

    draw.text((x, y), text, font=font, fill="white")

    return image

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_grid_from_lists(img_list, img_max_list=None, eval_list=None, title_list=None, zoom_region = None, zoom_size=None,
                         crop_vert=None, path=None, mag_factor=1.0, set_vmax=None, cmap="gray", cbar=False):

    cbar_dist = 0 if cbar is False else -2.5
    cbar_no = 0 if cbar is False else 1

    no_rows = len(img_list)
    no_columns = len(img_list[0]) + cbar_no

    if crop_vert is not None:
        crop_factor = (img_list[0][0].shape[0] - (2 * (crop_vert))) / img_max_list[0][0] if crop_vert is not None else 1.
        fig, ax = plt.subplots(no_rows, no_columns, figsize=(no_columns * 10 + cbar_dist, no_rows * 10))
    else:
        crop_factor = 1
        if img_list[0][0].shape[-1] == img_list[0][0].shape[-2]:
            fig, ax = plt.subplots(no_rows, no_columns, figsize=(no_columns * 10 + cbar_dist, no_rows * 10))
        else:
            ratio = img_list[0][0].shape[-2] / img_list[0][0].shape[-1]
            fig, ax = plt.subplots(no_rows, no_columns, figsize=(no_columns * 10 + cbar_dist, no_rows * 10 * ratio))

    for row in range(no_rows):
        for col in range(no_columns) if cbar is False else range(no_columns - 1):
            ax_i = ax[row, col] if no_rows > 1 else ax[col]
            if img_list[row][col] is not None:
                # img = medutils.visualization.contrastStretching(img_list[row,col])
                img = img_list[row][col] * mag_factor
                if set_vmax is not None:
                    vmax = set_vmax
                else:
                    vmax = img_max_list[row][col] if img_max_list is not None else img.max()
                if crop_vert is not None:
                    im = ax_i.imshow(img[crop_vert:-crop_vert,:], cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax) # vmin=0, vmax=l*img.max())
                else:
                    im = ax_i.imshow(img[:,:], cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax)# vmin=0, vmax=l*img.max())

                if zoom_region is not None:
                    # Add zoomed-in region
                    zoom_size = 0.35 if zoom_size is None else zoom_size
                    zoom_in = img[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]]
                    zoomed_ax = ax_i.inset_axes([1-zoom_size, 0.02, zoom_size, zoom_size*(1/crop_factor)],
                                                        transform=ax_i.transAxes)
                    zoomed_ax.imshow(zoom_in, cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax)
                    zoomed_ax.axis('off')

                    # Add a light white box around the zoomed-in region in the first image
                    if row == 0 and col == 0:
                        crop_shift = 0 if crop_vert is None else crop_vert
                        rect = patches.Rectangle((zoom_region[2], zoom_region[0]-crop_shift), zoom_region[3] - zoom_region[2],
                                                 zoom_region[1] - zoom_region[0], linewidth=2, edgecolor='white', alpha=1,
                                                 facecolor='none')
                        ax_i.add_patch(rect)

            ax_i.axis('off')

            if row == 0 and title_list is not None:
                ax_i.set_title(title_list[row][col], fontsize=16, c='white')

            if eval_list is not None:
                if eval_list[row] is not None:
                    if eval_list[row][col] is not None:
                        ax_i.text(0.05, 0.95, str(eval_list[row][col]), color='white', fontsize=40,
                                      horizontalalignment='left', verticalalignment='top',
                                      transform=ax_i.transAxes)

        if cbar:
            ax_i = ax[row, no_columns-1] if no_rows > 1 else ax[no_columns-1]
            cbar_ax = inset_axes(ax_i, width="40%", height="100%", loc='center',
                                 bbox_to_anchor=(0.0, 0.05, 0.2, 0.9), # x0,y0,width,height
                                 bbox_transform=ax_i.transAxes)  # Adjust width and height as needed

            cbar0 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax)), cax=cbar_ax)
            cbar0.ax.tick_params(labelsize=30)
            ax_i.axis('off')

            # Adjust spacing between subplots to zero
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            plt.tight_layout()
        else:
            plt.tight_layout()

        try:
            plt.show()
        except:
            pass

        if path is not None:
            fig.savefig(path, transparent=True, bbox_inches='tight', facecolor='none')

def plot_grid_w_eval(img_list_dict, eval_list_dict, title_list, results_path, perc=99, zoom = False):
    no_rows = len(img_list_dict[0])
    no_columns = len(img_list_dict)

    if img_list_dict[0][0].shape[-1] == img_list_dict[0][0].shape[-2]:
        fig, ax = plt.subplots(no_rows, no_columns, figsize=(no_rows * 10, no_columns * 10))
    else:
        ratio = img_list_dict[0][0].shape[-1] / img_list_dict[0][0].shape[-2]
        fig, ax = plt.subplots(no_rows, no_columns, figsize=(no_rows * 10, no_columns * 10 * ratio))

    for i, key in enumerate(img_list_dict[0].keys()):
        for j, col in enumerate(range(no_columns)):
            if img_list_dict[col][key] is not None:
                img_zf = np.abs(img_list_dict[col][key])
                img = medutils.visualization.normalize(img_zf)
                l = np.percentile(img, perc)
                ax[i, j].imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=l)

                if img_list_dict[col][key] is not None:
                    img_zf = np.abs(img_list_dict[col][key])
                    img = medutils.visualization.normalize(img_zf)
                    l = np.percentile(img, perc)
                    ax[i, j].imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=l)

                if zoom:
                    # Add zoomed-in region
                    zoom_region = img[150:225, 75:150]
                    zoomed_ax = ax[i, j].inset_axes([0.6, 0.02, 0.4, 0.4], transform=ax[i, j].transAxes)
                    zoomed_ax.imshow(zoom_region, cmap='gray', interpolation='nearest')
                    zoomed_ax.axis('off')

            else:
                pass
            ax[i, j].axis('off')

            if i == 0:
                ax[i, j].set_title(title_list[j])

            if key in eval_list_dict[col]:
                value = [eval_list_dict[col][key]["ssim"], eval_list_dict[col][key]["psnr"],
                         eval_list_dict[col][key]["nrmseAbs"]]
                ax[i, j].text(0.95, 0.95, str(value), color='white', fontsize=35,
                              horizontalalignment='right', verticalalignment='top',
                              transform=ax[i, j].transAxes)


    fig.tight_layout()
    fig.show()
    fig.savefig(results_path, transparent=True, bbox_inches='tight')

def to_uppercase(lst):
    return [s.upper() for s in lst]



from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
def plot_3d_slices_from_dict(array_dict, x=None, y=None, t=None, cmap='gray', results_path="",
                             vmax=None, stretch=False, eval_str_dict=None, fontsize=12,
                             zoom_region=None, zoom_size=None):
    """
    Plot cuts through 3D numpy arrays stored in a dictionary for specified x, y, or t.
    Optionally display dimension-specific strings in the top-left corner of each plot.

    Parameters:
    - array_dict: dictionary of 3D numpy arrays, keys are titles for the columns
    - x: int or None, index for the x-axis (optional)
    - y: int or None, index for the y-axis (optional)
    - t: int or None, index for the t-axis (optional)
    - eval_str_dict: dictionary of strings for dimensions to be shown on each subplot (optional)
    """
    # Determine dimensions
    dimensions = [t, y, x]
    labels = ['t', 'y', 'x']
    active_dims = [i for i, dim in enumerate(dimensions) if dim is not None]
    num_rows = len(active_dims)
    num_cols = len(array_dict)
    if stretch == False:
        vmax = vmax if vmax is not None else max([np.max(array) for array in array_dict.values()])
    else:
        import utils
        vmax_list = []
        for array in array_dict.values():
            _, vmin, vmax = utils.eval.contrastStretching(array, saturated_pixel=0.015)  # returns im in range 0-255, but vmin/max in original range
            vmax_list.append(vmax)
        vmax = max(vmax_list)

    # Get the arrays and their corresponding titles
    arrays = list(array_dict.values())

    # Determine aspect ratios of rows
    aspect_ratios = []
    for dim in active_dims:
        if labels[dim] == 't':
            aspect_ratios.append(arrays[0].shape[1] / arrays[0].shape[2])  # x:y
        elif labels[dim] == 'y':
            aspect_ratios.append(arrays[0].shape[0] / arrays[0].shape[1])  # x:z
        elif labels[dim] == 'x':
            aspect_ratios.append(arrays[0].shape[0] / arrays[0].shape[2])  # y:z

    # Calculate grid ratios based on aspect ratios
    total_height = sum(aspect_ratios)
    row_heights = [ar / total_height for ar in aspect_ratios]

    # Dynamically adjust figure size based on number of columns and row aspect ratios
    fig_width = 4 * num_cols  # Adjusting the width based on the number of columns
    fig_height = 4 * total_height  # Adjusting the height based on the row aspect ratios

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(fig_width, fig_height))  # Adjust figure size dynamically
    gs = GridSpec(num_rows, num_cols, height_ratios=row_heights, hspace=0.01, wspace=0.01)  # Minimize spacing

    # Normalize for consistent colormap scaling
    norm = Normalize(vmin=0, vmax=vmax)
    # Plot slices
    for col, (title, array) in enumerate(array_dict.items()):
        for row, dim in enumerate(active_dims):
            ax = fig.add_subplot(gs[row, col])

            # Select slice
            if labels[dim] == 't':
                slice_data = array[dimensions[dim], :, :]
            elif labels[dim] == 'y':
                slice_data = array[ :, dimensions[dim], :]
            elif labels[dim] == 'x':
                slice_data = array[:, :, dimensions[dim]]

            # Plot slice
            ax.imshow(slice_data, vmin=0, vmax=vmax, cmap=cmap)
            # if row == 0:
            #     ax.set_title(title, fontsize=fontsize)
            ax.axis('off')

            # Add zoomed-in region for 'xy' slices
            if labels[dim] == 't' and zoom_region is not None:
                # Extract zoomed-in region
                zoom_in = slice_data[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]]

                # Create an inset axis for the zoomed region
                zoomed_ax = ax.inset_axes(
                    [1 - zoom_size, 0.02, zoom_size, zoom_size * (zoom_in.shape[0] / zoom_in.shape[1])])
                zoomed_ax.imshow(zoom_in, cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax)
                zoomed_ax.axis('off')

                # Add a rectangle to indicate the zoomed region
                if row == 0 and col == 0:
                    rect = patches.Rectangle(
                        (zoom_region[2], zoom_region[0]),  # (x, y) bottom-left corner
                        zoom_region[3] - zoom_region[2],  # Width
                        zoom_region[1] - zoom_region[0],  # Height
                        linewidth=2, edgecolor='white', alpha=1, facecolor='none'
                    )
                    ax.add_patch(rect)

            if row == 0:
                ax.set_title(title, fontsize=fontsize)

            # Plot the line on the first image in the first row if a slice is taken
            if row == 0 and col == 0:
                if x is not None:
                    # Plot vertical line at x slice
                    ax.axvline(x=dimensions[-1], color='white', linestyle='--', linewidth=2)
                if y is not None:
                    # Plot horizontal line at y slice
                    ax.axhline(y=dimensions[-2], color='white', linestyle='--', linewidth=2)

            # Display dimension-specific string if provided
            if eval_str_dict is not None:
                # For each subplot, check if the string for the current dimension is provided
                dim_label = labels[dim]  # e.g., 'x', 'y', 't'
                if dim_label in eval_str_dict:
                    if title in eval_str_dict[dim_label]:
                        ax.text(0.95, 0.95, eval_str_dict[dim_label][title], transform=ax.transAxes,
                            color='white', fontsize=fontsize, va='top', ha='right', weight='bold')

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

    # Show the figure
    plt.show()

    # Save the figure
    if results_path:
        fig.savefig(results_path, transparent=True, bbox_inches='tight')

def plot_violin_plots(data, factors=None, metrics=None, models=None, model_name="PISCO-NIK", results_path="", font_size=16):
    factors = ["R1", "R2", "R3"] if factors is None else factors
    metrics = ["psnr", "rmse", "fsim", "fsim_xt", "fsim_yt"] if metrics is None else metrics
    models = ["xdgrasp4", "nik", "model"] if models is None else models

    # Define colors for models
    # colors = {"xdgrasp4": "black", "xdgrasp50": "gray", "nik": "#8EA5CE", "model": "#0059A0"}
    colors = {"xdgrasp4": "lightgray", "xdgrasp50": "darkgray", "nufft25": "lightgray", "xdgrasp25": "darkgray",
              "nik": "#8EA5CE", "ICoNIK": "#486AA6", "PISCO-dist": "#A9C09A", "PISCO": "#6C8B57"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics)*7, 5))  # Create one row of subplots

    for j, metric in enumerate(metrics):
        min_y = float("inf")
        max_y = float("-inf")
        for model in models:
            for factor in factors:
                metric_data = data[factor][model][metric]
                min_y = min(min_y, min(metric_data))
                max_y = max(max_y, max(metric_data))
        min_y = min_y * 0.9
        max_y = max_y * 1.1

        ax = axes[j]  # Use the same subplot for all factors
        handles = []
        labels = []
        group_width = 0.3  # Adjust the width of each group of violin plots
        group_gap = 0.1  # Adjust the gap between groups of violin plots
        factor_gap = 0.2
        total_width = len(models) * (group_width + group_gap)

        ## for each factor
        for i, factor in enumerate(factors):
            positions = []

            # plot all models
            for k, model in enumerate(models):
                metric_data = data[factor][model][metric]
                # Calculate positions for each model within the current factor's group
                position = [k * (group_width + group_gap) + i * (total_width + factor_gap)]
                positions.extend(position)
                parts = ax.violinplot(
                    metric_data, positions=position, showmeans=True, showextrema=False, widths=group_width,

                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(colors[model])
                    pc.set_alpha(1)
                parts["cmeans"].set_edgecolor("black")
                if i == 0:  # Add handles and labels only once
                    handles.append(Patch(facecolor=colors[model], label=model))
                    if model == "model":
                        labels.append(model_name.upper())
                    elif model == "PISCO-dist":
                        labels.append("PISCO-dist")
                    else:
                        labels.append(model.upper())

            # Statistical testing with Wilcoxon rank-sum test
            print("Testing for metric {} for factor {}".format(metric, factor))
            p_comb, p_values = statistical_testing(models, {model: data[factor][model][metric] for model in models})
            top = True if metric in ["rmse"] or "lpips" in metric else False
            show_brackets(np.array(p_values), np.array(p_comb), violins=np.array(positions),
                          height=20, ax=ax, top=top)#, ylim=[min_y, max_y])

        ax.set_ylim(min_y, max_y)
        if metric == "fsim_t":
            ax.set_ylabel("FSIM-temp", fontsize=font_size, fontweight='bold')
        elif metric == "fsim":
            ax.set_ylabel("FSIM-spat", fontsize=font_size, fontweight='bold')
        else:
            ax.set_ylabel(metric.upper(), fontsize=font_size, fontweight='bold')

        ax.set_xlabel("Acceleration Factor", fontsize=font_size, labelpad=15)  # Add bold xlabel in the middle

        ax.set_xticks(
            [i * (total_width + factor_gap) + (total_width /2) for i in range(len(factors))]
        )
        ax.set_xticklabels(factors, fontsize=font_size)  # Set factor names as x-axis ticks
        if metric == "psnr":
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))  # Limit y-axis to 2 points after comma
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=font_size)
        ax.grid(axis="y", color='lightgray', linestyle='--', linewidth=0.5)  # Add gray horizontal grid lines

        # Adjust y-axis limits to match the gray lines
        yticks = ax.get_yticks()
        min_y_tick = yticks[0] # - 0.05 * (yticks[-1] - yticks[0])
        max_y_tick = yticks[-1] # + 0.05 * (yticks[-1] - yticks[0])
        ax.set_ylim(min_y_tick, max_y_tick)
        ax.tick_params(axis="y", labelsize=font_size)


        # Legend
        if j == 0:
            legend_ax = fig.add_subplot(111, frameon=False)
            legend_ax.axis("off")
            legend_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1), fontsize=font_size)

    # fig.set_xlabel("Acceleration Factor")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)  # Adjust the spacing between subplots
    fig.show()
    fig.savefig(results_path, transparent=True, bbox_inches="tight")
    plt.show()

def show_brackets(p_cor, ind, violins, height, col='dimgrey', top=True,
                  p_value_threshold=0.05, ax=None, ylim=None):
    """Show brackets with 'N' for p_values larger than p_value_threshold.

    Parameters
    ----------
    p_cor : np.array
        Corrected p-values.
    ind : np.array
        Indices of the violins to annotate.
    violins : np.array
        Violin plot positions.
    height : float
        Height of the brackets.
    col : str
        Color of the brackets.
    top : bool
        Whether to show the brackets above the violins.
    p_value_threshold : float
        Threshold for p-values to show the brackets.
    ax : matplotlib.axes.Axes
        Axes to plot the brackets on.
    ylim : list
        Y-axis limits of the plot.
    """

    # Calculate the height of the plot for scaling
    if ylim is not None:
        plot_height = ylim[1] - ylim[0]
        height = 1.01 * ylim[1] if top else 0.99 * ylim[0]
    else:
        plot_height = ax.get_ylim()[1] - ax.get_ylim()[0]
        height = 1.01 * ax.get_ylim()[1] if top else 0.99* ax.get_ylim()[0]

    ind = ind[p_cor >= p_value_threshold]
    heights = [height for s in ind]
    previous_brackets = []
    for i in range(len(ind)):
        if ind[i][0] in previous_brackets or ind[i][1] in previous_brackets:
            if top:
                heights[i] += 0.12*plot_height
            else:
                heights[i] -= 0.12*plot_height
        previous_brackets.append(ind[i][0])
        previous_brackets.append(ind[i][1])

    # Calculate the height of the bracket as a ratio of the plot height
    bracket_heights = 0.02 * plot_height

    for i in range(len(ind)):
        barplot_annotate_brackets(
            violins[ind[i][0]], violins[ind[i][1]],  heights[i],
            bracket_heights, col, 13, top, label="N", ax=ax
        )

def barplot_annotate_brackets(num1, num2, y, bracket_height,
                              col, fs, top, label="", ax=None):
    """Annotate bar plot with brackets and text."""
    lx, ly = num1, y
    rx, ry = num2, y

    if top:
        bary = [y, y + bracket_height, y + bracket_height, y]
        text_position = y + bracket_height
        va = 'bottom'
    else:
        bary = [y + bracket_height, y, y, y + bracket_height]
        text_position = y - bracket_height
        va = 'top'

    ax.plot([lx, lx, rx, rx], bary, c=col)
    ax.text((lx + rx) / 2, text_position, label, ha='center', va=va,
             fontsize=fs, color=col)

import itertools
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
def statistical_testing(keys, metric):
    """
    Perform statistical testing using Wilcoxon signed rank tests and multiple
can    comparison correction (FDR) on metric values for pairs of keys.

    Parameters
    ----------
    keys : list
        Sorted keys of the metrics dictionary.
    metric : dict
        Dictionary containing metric values for each key.

    Returns
    -------
    tuple
        A tuple containing combinations and p-values for statistical testing.
    """

    combinations = list(itertools.combinations(np.arange(0, len(keys)),
                                               2))
    p_values = []
    for comb in combinations:
        p_values.append(wilcoxon(metric[keys[comb[0]]],
                                 metric[keys[comb[1]]],
                                 alternative='two-sided')[1])
    rej, p_values_cor, _, __ = multipletests(p_values, alpha=0.05,
                                             method='fdr_bh', is_sorted=False,
                                             returnsorted=False)

    insign = np.where(p_values_cor >= 0.05)
    for ins in insign[0]:
        print(keys[combinations[ins][0]],
              keys[combinations[ins][1]],
              " No significant difference.")

    return combinations, p_values



def plot_violin_plots2(data, factors = None, metrics = None, models = None, model_name="k-Reg", results_path = ""):
    factors = ['R1','R2', 'R3'] if factors is None else factors
    metrics = ['psnr', 'rmse', 'fsim', 'fsim_xt', 'fsim_yt'] if metrics is None else metrics
    models = ["xdgrasp4", "nik", "model"] if models is None else models

    # Define colors for models
    colors = {'xdgrasp4': 'gray',
              'nik': '#8EA5CE',
              'model': '#0059A0'}

    fig, axes = plt.subplots(len(factors), len(metrics), figsize=(15, 10))

    for j, metric in enumerate(metrics):
        min_y = float('inf')
        max_y = float('-inf')
        for i, factor in enumerate(factors):
            for model in models:
                metric_data = data[factor][model][metric]
                min_y = min(min_y, min(metric_data))
                max_y = max(max_y, max(metric_data))
                min_y = min_y * 0.9995
                max_y = max_y * 1.0005

        for i, factor in enumerate(factors):
            ax = axes[i, j]
            handles = []
            labels = []
            metric_data_all_models = []
            for model in models:
                metric_data = data[factor][model][metric]
                metric_data_all_models.append(metric_data)
                parts = ax.violinplot(metric_data, positions=[models.index(model) + 1], showmeans=True, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[model])
                parts["cmeans"].set_edgecolor('gray')
                handles.append(Patch(facecolor=colors[model], label=model))
                if model == "model":
                    labels.append(model_name)
                else:
                    labels.append(model)

            ax.set_ylim(min_y, max_y)
            ax.set_xlabel('')
            if i == 0:
                ax.set_title(f'{metric}')
            else:
                ax.set_title('')
            if i == len(factors) - 1:
                ax.set_xticks(np.arange(1, len(models) + 1))
                ax.set_xticklabels(labels)  # Set model names as x-axis ticks
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # Limit y-axis to 2 points after comma
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Set maximum of 5 ticks on y-axis
            ax.tick_params(axis="x", which="both", bottom=False, top=False)
            ax.grid(axis="y", linestyle="--", color="gray")  # Add gray horizontal grid lines

    # Legend
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 1))

    fig.tight_layout()
    fig.show()
    fig.savefig(results_path, transparent=True, bbox_inches='tight')
    plt.show()



def plot_slices(img_3d, slicedim=[1,2], timedim=10, indices=None, save_path=None):

    if indices is None:
        indices = np.arange(20, img_3d.shape[1] -20, 20)

    M, N = len(indices), 1

    img_slices_1 = np.moveaxis(img_3d, slicedim[0], 0) # move slice index to front
    img_slices_1 = np.take(img_slices_1, indices, axis=0)

    img_slices_2 = np.moveaxis(img_3d, slicedim[1], 0) # move slice index to front
    img_slices_2 = np.take(img_slices_2, indices, axis=0)

    img_clean = np.take(img_3d, timedim, axis=0).copy()
    img_marked = np.take(img_3d, timedim, axis=0).copy()

    for slice_idx in slicedim:
        for i, index in enumerate(indices):
            index_array = [slice(None)] * len(img_marked.shape)
            index_array[slice_idx-1] = index
            img_marked[tuple(index_array)] = img_marked.max() # fill the index line with red

    # Plot the slices with lines of indices
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # Remove grid from all subplots
    for ax_row in ax:
        for ax_col in ax_row:
            ax_col.grid(False)
            ax_col.axis('off')
    ax[0,0].imshow(contrastStretching(img_clean), cmap='gray')
    ax[0,0].set_title("t={}".format(timedim))
    ax[0,1].imshow(contrastStretching(plot_array(img_slices_2, M=M, N=N).T),'gray')
    ax[0,1].set_title("yt")
    ax[1,0].imshow(contrastStretching(plot_array(img_slices_1, M=M, N=N)),'gray')
    ax[1,0].set_title("xt")
    ax[1,1].imshow(contrastStretching(img_marked),'gray')
    ax[1,1].set_title("Marked intersections")

    fig.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path)

def rgba_to_grayscale(rgba_array):
    # Extract the R, G, and B channels from the RGBA array
    r_channel = rgba_array[:, :, 0]
    g_channel = rgba_array[:, :, 1]
    b_channel = rgba_array[:, :, 2]

    # Calculate the grayscale value using the formula: 0.299*R + 0.587*G + 0.114*B
    grayscale_array = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel

    return grayscale_array

def plot_magnitude_weighting(image, traj, title=None, r=1, p=1, eps=0.01, sigma=1, cmap="Blues"):
    """
    Expected input shape: coils * x * y array with magnitude values
    """

    assert len(image.shape) == 3 #
    assert image.shape[1:] == traj.shape[:-1]
    assert traj.shape[-1] == 2

    size = image.shape
    b = size[0]
    x = np.arange(size[1])
    y = np.arange(size[2])

    image_reshaped = image.reshape(size[0]*size[1], size[2])
    dist_to_center = traj[...,0]**2 + traj[...,1]**2

    xx, yy = np.meshgrid(np.arange(image_reshaped.shape[0]), y, indexing='ij')
    # xx = xx - size[0] // 2
    # yy = yy - size[1] // 2

    # generate weight map
    # weight = np.power(r * (xx ** 2 + yy ** 2), p) + eps

    ## frequency regulariation loss
    # weight = (1 -  np.exp(-(dist_to_center)/(2 * sigma**2)))
    # weight_string = "(1 -  e^2(-(r)/(2 * {}**2)))".format(sigma)
    # weight = weight.repeat(b,0)     # stack first dim over x axis

    # hdr loss
    # max_value = 0.7
    # image_norm = image/ np.max(image) * max_value
    weight = 0.1 / (image + eps) # assuming error is 0.1, plot the loss that results out of that
    weight_string = "( 0.1  / (kdata + {}))".format(eps)
    weight = weight.reshape(-1, size[2])

    # xx = xx.repeat(b,0)
    # yy = yy.repeat(b,0)
    weighted_kspace = weight * image_reshaped
    fig = plt.figure()
    # set size of the figure
    fig.set_size_inches(18.5, 6.5)
    # set margin of the figure
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    ax = fig.add_subplot(131, projection='3d', title='weight: {}'.format(weight_string))
    ax.plot_surface(xx, yy, weight, cmap=cmap)
    # lim = weight[0,1]
    # ax.set_zlim(0, lim)

    ax = fig.add_subplot(132, projection='3d', title='kspace')
    ax.plot_surface(xx, yy, np.abs(image_reshaped), cmap=cmap)
    ax.set_zlim(0, 1)

    ax = fig.add_subplot(133, projection='3d', title='kspace*weight')
    ax.plot_surface(xx, yy, np.abs(weighted_kspace), cmap=cmap)
    ax.set_zlim(0, 0.01)
    # plt.show()
    plt.savefig("kspace_distribution_echo2_{}.png".format(title), transparent=True)
    plt.show()

def plot_kreg_weights(W, wandb_log=False):

    if isinstance(W, torch.Tensor):
        W = W.cpu().numpy()
    else:
        W = W
    W = W.reshape(W.shape[0], -1)

    # Separate real and imaginary parts.
    real_parts = np.real(W)
    imaginary_parts = np.imag(W)
    mag_parts = np.abs(W)
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))

    # Plot real parts
    axes[0].plot(real_parts.T, alpha=0.7)
    axes[0].set_title('Real Parts of Weights')
    axes[0].set_xlabel('Weight Index')
    axes[0].set_ylabel('Real Part')

    # Plot imaginary parts
    axes[1].plot(imaginary_parts.T, alpha=0.7)
    axes[1].set_title('Imaginary Parts of Weights')
    axes[1].set_xlabel('Weight Index')
    axes[1].set_ylabel('Imaginary Part')

    # Plot magnitude parts
    axes[2].plot(mag_parts.T, alpha=0.7)
    axes[2].set_title('Magnitude of Weights')
    axes[2].set_xlabel('Weight Index')
    axes[2].set_ylabel('Magnitude')

    fig.tight_layout()
    img_path = 'weights_plot_3.png'
    fig.savefig(img_path)
    plt.close()

    if wandb_log:
        wandb_dict = {"Weights_Plot": wandb.Image(img_path)}
        # Log the image to WandB
        return wandb_dict

def plot_mag_and_phase(data, title, vmin=None, vmax=None, log_wandb=False, frames=None, fontsize=12):
    """
    Plot the magnitude and phase of a complex data slice.

    Parameters:
    data_slice (torch.Tensor or np.ndarray): Input with shape (..., 2) ([...,0] for magnitude, [...,1] for phase).
    title (str): Title for the plot.
    vmin (float): Minimum value for magnitude color scale.
    vmax (float): Maximum value for magnitude color scale.
    log_wandb (bool): Whether to log images to wandb.
    frames (list): List to append wandb images if logging.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Create a grid with subplots and space for colorbars
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [10, 10], 'wspace': 0.05})

    # Plot magnitude
    mag_im = axs[0].imshow(data[..., 0], cmap='viridis',
                           aspect='auto',
                           vmin=vmin, vmax=vmax)
    axs[0].set_title('Magnitude', fontsize=fontsize)
    axs[0].set_xlabel('Weight index', fontsize=fontsize)
    axs[0].set_ylabel('Iteration index', fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize*0.75)

    # Remove y-axis from second plot
    phase_im = axs[1].imshow(data[..., 1], cmap='twilight_shifted',
                             aspect='auto',
                             vmin=-np.pi, vmax=np.pi)
    axs[1].set_title('Phase', fontsize=fontsize)
    axs[1].set_xlabel('Weight index', fontsize=fontsize)
    axs[1].set_yticks([])  # Remove y-axis ticks
    axs[1].set_ylabel('')  # Remove y-axis label
    axs[1].tick_params(labelsize=fontsize*0.75)

    # Create a single axes for the colorbars on the right
    cbar_ax_mag = fig.add_axes([0.91, 0.53, 0.02, 0.35])  # For magnitude colorbar
    cbar_ax_phase = fig.add_axes([0.91, 0.11, 0.02, 0.35])  # For phase colorbar

    # Add colorbars to the right of the plots
    mag_cbar = fig.colorbar(mag_im, cax=cbar_ax_mag)
    # mag_cbar.set_label('Magnitude', fontsize=fontsize)
    mag_cbar.ax.tick_params(labelsize=fontsize*0.75)  # Set colorbar tick label font size
    if vmax is not None:
        mag_cbar.set_ticks([0.0, vmax/2, vmax])  # Set specific ticks

    phase_cbar = fig.colorbar(phase_im, cax=cbar_ax_phase)
    # phase_cbar.set_label('Phase (radians)', fontsize=fontsize)
    phase_cbar.ax.tick_params(labelsize=fontsize*0.75)  # Set colorbar tick label font size
    phase_cbar.set_ticks([-np.pi, 0, np.pi])  # Set specific ticks for phase
    phase_cbar.set_ticklabels(['-π', '0', 'π'])  # Set custom tick labels

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbars on the right

    if log_wandb:
        image = wandb.Image(fig)
        frames.append(image)
        return frames

    else:
        fig.savefig(f"{title}.png", transparent=True)
        fig.savefig(f"{title}.eps", transparent=True)
        fig.savefig(f"{title}.pdf", transparent=True)
        plt.show()

    plt.close(fig)

def log_kspace_weights(W, title="weights", vmin=0, vmax=1, log_wandb=True):
    wandb_dict = {}
    frames = []
    if isinstance(W, torch.Tensor):
        W = W.cpu().numpy()
    else:
        W = W

    # vmax = np.percentile(W, 99)

    if W.ndim == 3:  # for case the mean weights are inputted
        data_slice = W
        if log_wandb:
            frames = plot_mag_and_phase(data_slice, title, vmin=vmin, vmax=vmax, log_wandb=log_wandb, frames=frames)
        else:
            plot_mag_and_phase(data_slice, title, vmin=vmin, vmax=vmax, log_wandb=log_wandb, frames=frames, fontsize=18)
        # fig, axs = plt.subplots(1,2, figsize=(12,4))
        # fig.suptitle(title)
        # im = axs[0].imshow(data_slice[...,0], cmap='viridis',
        #                 aspect='auto',
        #                 # extent=[0, data_slice.shape[0], 0, data_slice.shape[1]],
        #                 vmin=vmin, vmax=vmax)
        # axs[0].set_title('mean weights[')
        # axs[0].set_xlabel('Weight index')
        # axs[0].set_ylabel('Iteration index')
        # cbar = fig.colorbar(im, ax=axs[0])
        #
        # im = axs[1].imshow(data_slice[...,1], cmap='twilight_shifted', # cyclic colormap for phase
        #                 aspect='auto',
        #                 # extent=[0, data_slice.shape[0], 0, data_slice.shape[1]],
        #                 vmin=-np.pi, vmax=np.pi)
        # axs[1].set_title('mean weights')
        # axs[1].set_xlabel('Weight index')
        # # axs[1].set_ylabel('Iteration index')
        # cbar = fig.colorbar(im, ax=axs[1])
        # fig.tight_layout()
        #
        # if log_wandb:
        #     image = wandb.Image(im)
        #     frames.append(image)
        # else:
        #     fig.savefig(title+".png")
        #     plt.show()
        # plt.close()

        # Log the Matplotlib figure to WandB as an image
        if log_wandb:
            wandb_dict.update({'{}'.format(title): frames})
            return wandb_dict

    elif W.shape[1] > 1:
        for i in range(W.shape[0]):  # Assuming you have 30 frames
            # Example data
            # if len(W.shape) == 3:
            #     W = np.stack([W,np.zeros_like(W)], axis=-1)
            data_slice = W[i, :, :]

            if log_wandb:
                frames = plot_mag_and_phase(data_slice, title, vmin=vmin, vmax=vmax, log_wandb=log_wandb, frames=frames)
            else:
                plot_mag_and_phase(data_slice, title, vmin=vmin, vmax=vmax, log_wandb=log_wandb, frames=frames, fontsize=18)
            # fig, axs = plt.subplots(1,2, figsize=(12,4))
            #
            # # Create Matplotlib figure with imshow
            # fig, axs = plt.subplots(1,2, figsize=(12,4))
            # fig.suptitle(title + f' - Iteration {i}')
            # im = axs[0].imshow(data_slice[...,0], cmap='viridis',
            #                 aspect='auto',
            #                 # extent=[0, data_slice.shape[0], 0, data_slice.shape[1]],
            #                 vmin=vmin, vmax=vmax)
            # axs[0].set_title('Iteration {}'.format(i))
            # axs[0].set_xlabel('Weight index')
            # axs[0].set_ylabel('Set index')
            # cbar = fig.colorbar(im, ax=axs[0])
            #
            # im = axs[1].imshow(data_slice[...,1], cmap='twilight_shifted',
            #                 aspect='auto',
            #                 # extent=[0, data_slice.shape[0], 0, data_slice.shape[1]],
            #                 vmin=-np.pi, vmax=np.pi)
            # axs[1].set_title('Iteration {}'.format(i))
            # axs[1].set_xlabel('Weight index')
            # # axs[1].set_ylabel('Set index')
            # cbar = fig.colorbar(im, ax=axs[1])
            # plt.title('Iteration {}'.format(i))
            # fig.tight_layout()
            #
            # # Log the Matplotlib figure to WandB as an image
            # if log_wandb:
            #     image = wandb.Image(im)
            #     frames.append(image)
            # else:
            #     fig.savefig(title + ".png")
            #     plt.show()
            # plt.close()

        # Log the Matplotlib figure to WandB as an image
        if log_wandb:
            wandb_dict.update({'{}'.format(title): frames})
            return wandb_dict

def plot_coords_scatter(data, data_patch = None, dim_order=[1,2,0], axis_name=["kx", "ky", "t"]):
    '''
    data: [sets, samples, coord]
    data_patch: [sets, samples, neighbors, coord]
    '''
    assert data.shape[-1] == 3
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Extract the three dimensions
    x = data[..., dim_order[0]]
    y = data[..., dim_order[1]]
    z = data[..., dim_order[2]]

    if data_patch is not None:
        # Extract the three dimensions for the patch points
        patch_x = data_patch[..., dim_order[0]]
        patch_y = data_patch[..., dim_order[1]]
        patch_z = data_patch[..., dim_order[2]]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], label=f'Subset {i + 1}')
        if data_patch is not None:
            for j in range(patch_x.shape[-2]):
                box_points = np.column_stack([patch_x[i, j, ...], patch_y[i, j, ...], patch_z[i, j, ...]])
                box_points = np.vstack([box_points, box_points[0, :]])  # Close the box
                ax.scatter(box_points[:, 0], box_points[:, 1], box_points[:, 2], color='red')

    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_zlabel(axis_name[2])
    ax.legend(loc='center right', bbox_to_anchor=(0, 0.5))
    fig.tight_layout()
    # fig.savefig("debug/kreg_coords_scatter.png")
    plt.show()


def plot_coords_scatter2D(data, data_patch = None, dim_order=[0,1], axis_name=["kx", "ky", "t"]):
    assert data.shape[-1] == 2
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Extract the three dimensions
    x = data[..., dim_order[0]]
    y = data[..., dim_order[1]]

    if data_patch is not None:
        # Extract the three dimensions for the patch points
        patch_x = data_patch[..., dim_order[0]]
        patch_y = data_patch[..., dim_order[1]]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(x)):
        ax.scatter(x[i], y[i], label=f'Subset {i + 1}')
        if data_patch is not None:
            for j in range(patch_x.shape[-2]):
                box_points = np.column_stack([patch_x[i, j, ...], patch_y[i, j, ...]])
                box_points = np.vstack([box_points, box_points[0, :]])  # Close the box
                ax.plot(box_points[:, 0], box_points[:, 1], color='red')

    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    # fig.savefig("debug/kreg_coords_scatter.png")
    plt.show()


def plot_PTD(P, T, D, cut=20):
    '''
    Expected input:
    P = [n_B, nSets, neighbors*coords]
    T = [n_B, nSets, coords]
    D = [n_B, nSets]
    '''
    if isinstance(P, np.ndarray):
        flag = True
        P, T, D = numpy2torch(P), numpy2torch(T), numpy2torch(D)

    import medutils.visualization as vis
    fig, axes = plt.subplots(1, 3, figsize=(6, 12))  # Create a figure with 1 row and 2 columns
    T_cut = T[0, :cut, :]
    P_cut = P[0, :cut, :]
    T_array = torch.abs(T_cut[...].reshape(cut, 1, -1)).detach().cpu().numpy()
    P_array = torch.abs(P_cut[...].reshape(cut, 1, -1)).detach().cpu().numpy()
    axes[0].imshow(vis.contrastStretching(vis.plot_array(P_array, N=1, M=cut)), aspect='auto')
    axes[0].set_title('P')
    axes[1].imshow(vis.contrastStretching(vis.plot_array(T_array, N=1, M=cut)), aspect='auto')
    axes[1].set_title('T')
    axes[2].imshow(D[[0], :cut].repeat(T_array.shape[-1], 1).permute(1, 0).cpu().numpy(),
                   aspect='auto')  # Flatten Dt_array if it's a 2D array
    axes[2].set_title('distance')
    plt.grid("off")
    plt.tight_layout()
    plt.show()

def plot_histogram_2d(data, min_data=0.0, max_data=1.0, bins=10, cmap='hot'):
    """
    Plots a 2D heatmap of the distributions for each row in a 2D array.

    Parameters:
    data : np.ndarray
        2D array of shape (n_rows, n_cols) containing the data.
    bins : int
        Number of bins for the histogram.
    cmap : str
        Colormap to use for the heatmap. Default is 'hot'.
    """
    n_rows, n_cols = data.shape

    # Create an array to hold histograms
    histograms = np.zeros((n_rows, bins))

    # Calculate bin edges based on the range of data and the number of bins
    bin_edges = np.linspace(min_data, max_data, bins + 1)
    # Calculate bin centers from the bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate histogram for each row
    for i in range(n_rows):
        # Calculate the histogram
        hist, bin_edges = np.histogram(data[i], bins=bins, range=(min_data, max_data))
        # Normalize the histogram
        histograms[i] = hist / np.sum(hist)

    # Plot the 2D heatmap
    plt.figure(figsize=(10, 8))
    c = plt.imshow(histograms, aspect='auto', cmap=cmap, interpolation='nearest',
                   extent=[bin_centers[0], bin_centers[-1], n_rows, 0])

    # Add a colorbar for the frequency
    plt.colorbar(c, label='Frequency')

    # Add labels and title
    plt.xlabel('Bin Index')
    plt.ylabel('Row Index')
    plt.title('2D Heatmap of Row Distributions')

    try:
        fig = plt.gcf()
        wandb.log({"patch_distribution": wandb.Image(fig)})
        plt.clf()
    except:
        # Show the plot
        plt.show()

def save_cuts(img, resultspath, title, data_type="cardiac_cine", extend_dim=None):

    img = torch2numpy(img)
    name_string = title

    if data_type == "cardiac_cine":
        cut_x = img.shape[-1] // 2
        cut_y = img.shape[-2] // 2
        saturated_pixel=0.00
    elif data_type == "abdominal_sos":
        cut_x = 75
        cut_y = 160
        saturated_pixel = .015
    elif data_type == "abdominal_phantom":
        saturated_pixel = 0.04
        # crop image
        img = center_crop(img, (img.shape[-2]//2, img.shape[-1]//2))
        cut_x = img.shape[-1] // 2
        cut_y = img.shape[-2] // 2

    img = contrastStretching(img, saturated_pixel=saturated_pixel)

    imsave(plot_array(img[0, 0, :, 0, ...]), os.path.join(resultspath, "grasp_{}".format(name_string)))

    ### Temporal cuts
    if extend_dim > img.shape[2]:
        if img.shape[2] == 1:
            img = np.repeat(img, extend_dim, axis=2)
        elif extend_dim == 50 and img.shape[2] == 4:
            repeat_counts = np.array([13, 12, 13, 12])
            img = np.concatenate([np.repeat(img[:, :, i:i + 1, ...], repeat_counts[i], axis=2)
                                  for i in range(len(repeat_counts))], axis=2)

    imsave((img[0, 0, :, 0, cut_y, ...]),
               os.path.join(resultspath, "grasp_xt_{}".format(name_string)))
    imsave((img[0, 0, :, 0, :, cut_x]),
           os.path.join(resultspath, "grasp_yt_{}".format(name_string)))

    save_gif(img[0, 0, :, ...], filename=os.path.join(resultspath, "grasp_xt_{}_animation.gif".format(name_string)))


def plot_kspace_comparison(all_kdata, overall_title, titles=None, diff_scale=5, path=None):
    """
    Plots a comparison of k-space data.
    Parameters:
    - all_kdata: List of k-space data arrays to plot.
    - acc_factor: Acceleration factor for the plot title.
    - titles: List of titles for each subplot (optional).
    - diff_scale: Scaling factor for the difference plot (default is 5).
    """
    all_kdata_log = [np.log(data+1e-4) for data in all_kdata]
    vmin = [data.min() for data in all_kdata_log]
    vmax = [data.max() for data in all_kdata_log]
    # Define default titles if not provided
    if titles is None:
        titles = ['Orig', 'Undersampled (Training data)', 'Optim',
                  '[Undersampled - Orig] x {}'.format(diff_scale),
                  '[Optim - Orig] x {}'.format(diff_scale),
                  '[Optim - Undersampled] x {}'.format(diff_scale), ]
    if len(all_kdata) == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, data in enumerate(all_kdata_log):
            axes[i].imshow(data, cmap='gray')
            axes[i].set_title(titles[i])
            axes[i].axis('off')  # Remove axes
        # Plot the difference
        diff = np.log(np.abs(all_kdata[1] - all_kdata[0]) +1e-4) * diff_scale
        axes[2].imshow(diff, cmap='gray')
        axes[2].set_title('[Undersampled - Orig] x {}'.format(diff_scale))
        axes[2].axis('off')  # Remove axes
    else:
        fig, axes = plt.subplots(1, 6, figsize=(18, 6))
        # Plot each k-space data
        for i, data in enumerate(all_kdata_log):
            axes[i].imshow(data, cmap='gray')
            axes[i].set_title(titles[i])
            axes[i].axis('off')  # Remove axes
        # Plot the difference if we have at least 3 datasets
        if len(all_kdata) >= 3:
            diff = np.abs(all_kdata[1] - all_kdata[0]) * diff_scale
            axes[3].imshow(diff, cmap='gray')
            axes[3].set_title(titles[3])
            axes[3].axis('off')  # Remove axes
            if all_kdata[2].shape == all_kdata[0].shape:
                diff = np.abs(all_kdata[2] - all_kdata[0]) * diff_scale
                axes[4].imshow(diff, cmap='gray')
                axes[4].set_title(titles[4])
                axes[4].axis('off')  # Remove axes
            if all_kdata[2].shape == all_kdata[1].shape:
                diff = np.abs(all_kdata[2] - all_kdata[1]) * diff_scale
                axes[5].imshow(diff, cmap='gray')
                axes[5].set_title(titles[5])
                axes[5].axis('off')  # Remove axes
    # Add an overall title
    fig.suptitle(overall_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if path is not None:
        plt.savefig(path)
    plt.show()

def plot_image_histograms(image_dict, percentile=[0, 1]):
    """
    Plots histograms of images stored in a dictionary using subplots with the same y-axis limits.

    Parameters:
        image_dict (dict): A dictionary where keys are image names (or IDs) and values are images (numpy arrays).
    """
    num_images = len(image_dict)  # Number of images to plot
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # Create subplots

    # If there is only one image, make axes iterable
    if num_images == 1:
        axes = [axes]

    # Store maximum count across all histograms to unify the y-axis
    max_hist_value = 0

    # First loop to find the maximum histogram value
    for i, (key, img) in enumerate(image_dict.items()):
        img = np.abs(img)
        image_dict[key] = clip_to_percentiles(img, lower=percentile[0], upper=percentile[1])

    # First loop to find the maximum histogram value
    # for i, (key, img) in enumerate(image_dict.items()):
    #     pixel_values = np.abs(img).flatten()
    #     hist, _ = np.histogram(pixel_values, bins=256, range=(0, 1))
    #     max_hist_value = max(max_hist_value, hist.max())


    # Second loop to plot the histograms with unified y-axis
    for i, (key, img) in enumerate(image_dict.items()):
        pixel_values = np.abs(img).flatten()
        axes[i].hist(pixel_values, bins=256, range=(0, 1), color='gray', alpha=0.7)
        axes[i].set_title(f'Histogram of {key}')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')

        # Set the same y-axis limits for all histograms
        # axes[i].set_ylim(0, max_hist_value)

    # Adjust layout
    plt.tight_layout()
    plt.show()
