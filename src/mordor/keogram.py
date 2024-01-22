
from PIL import Image, ImageDraw

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import trosat.sunpos as sp

def slice_setup(image, freq, radius=None):
    ## freq pics per minute
    img_width, img_height = image.size
    img_centerx, img_centery = img_width // 2, img_height // 2
    if radius is None:
        radius = img_centerx

    # width of final image, one columne of pixels for each image taken
    out_width = 24. * 60. * freq
    # required super sampling scale to achieve the desired width
    out_scale = out_width / (np.pi * radius)
    # angle of original picture corresponding to 1 pixel column in distorted picture
    out_dangle = 360. / out_width
    return out_scale, out_dangle, out_width

def concat_images(left_image, right_image):
    new_image = Image.new('RGB', (left_image.width + right_image.width, right_image.height))
    new_image.paste(left_image, (0, 0))
    new_image.paste(right_image, (left_image.width, 0))
    return new_image

def test_image_config(
    img_file,
    img_date,
    longitude=None,
    radius_scale=1,
    angle_offset=0,
    flip=False
):
    with Image.open(img_file) as cimage:
        draw = ImageDraw.Draw(cimage)
        radius = int(radius_scale * cimage.size[0] // 2)
        img_width, img_height = cimage.size
        img_centerx, img_centery = img_width // 2, img_height // 2
        draw.ellipse(
            [img_centerx-radius,img_centery-radius,
             img_centerx+radius,img_centery+radius],
            outline=(255, 0, 0),
            fill=None
        )

        if flip:
            # flip image if required
            cimage = cimage.transpose(Image.FLIP_LEFT_RIGHT)

        # rotate into the sun
        hangle = sp.hour_angle(img_date, lon=longitude, units=sp.units.DEG)
        image_out = cimage.rotate(angle_offset + hangle)
    return image_out

def make_keogram(
    img_files,
    img_dates,
    longitude=None,
    radius_scale=1,
    angle_offset=0,
    flip=False,
    fill_color=(0, 0, 0),
    whole_day=False
):
    keogram_image = Image.new("RGB", (0, 0))

    # analyze frequency of data and fill gaps
    freqs, counts = np.unique(np.diff(img_dates), return_counts=True)
    freq = freqs[np.argmax(counts)]
    img_freq = freq.seconds / 60.  # pics per minute

    # complete dates
    if whole_day:
        sdate = np.datetime64(img_dates[0]).astype("datetime64[D]")
        date_filled = pd.date_range(sdate, sdate + np.timedelta64(1, 'D'), freq=freq)
    else:
        date_filled = pd.date_range(img_dates[0], img_dates[-1], freq=freq)

    # analyze scale of slices
    with Image.open(img_files[0]) as cimage:
        radius = int(radius_scale * cimage.size[0] // 2)
        out_scale, out_dangle, out_width = slice_setup(cimage, freq=img_freq, radius=radius)

    # setup filling slice
    filling_slice = Image.new("RGB", (2 * int(1 / out_scale), 2 * radius), color=fill_color)

    for dt in date_filled:
        if dt not in img_dates:
            keogram_image = concat_images(keogram_image, filling_slice)
            continue

        with Image.open(img_files[img_dates.index(dt)]) as cimage:
            radius = int(radius_scale * cimage.size[0] // 2)

            if flip:
                # flip image if required
                cimage = cimage.transpose(Image.FLIP_LEFT_RIGHT)

            # rotate into the sun
            hangle = sp.hour_angle(dt, lon=longitude, units=sp.units.DEG)
            cimage = cimage.rotate(angle_offset + hangle)

            # crop to slice
            img_width, img_height = cimage.size
            img_centerx, img_centery = img_width // 2, img_height // 2
            slice = (img_centerx, img_centery - radius, img_centerx + 2 * int(1 / out_scale), img_centery + radius)
            cimage = cimage.crop(slice)

            # attach to keogram
            keogram_image = concat_images(keogram_image, cimage)
    return keogram_image


def plot_keogram(keogram_image, sdate, edate, ax=None, newfig=False):
    if newfig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if ax is None:
        ax = plt.gca()

    extent = [
        np.datetime64(sdate),
        np.datetime64(edate),
        0, keogram_image.size[1]
    ]
    ax.imshow(np.asarray(keogram_image), extent=extent)
    ax.axhline(keogram_image.size[1]//2, color='k', ls=':')
    ax.text(np.datetime64(sdate), keogram_image.size[1]//2, "zenith",
            va='center', ha='right', rotation=90)
    ax.set_aspect('auto')
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel("time (UTC)")
    ax.tick_params(axis='y', left=False, labelleft=False)
    return fig, ax