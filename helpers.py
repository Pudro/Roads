import os
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import random
import numpy as np


class SurfaceMapping:
    def __init__(self):
        self.mapping_dict = {
            'Alley': 1,
            'Paved Drive': 2,
            'Parking Lot': 3,
            'Road': 4,
            'Intersection': 5,
            'Paved Median Island': 6,
            'Paved Traffic Island': 7,
            'Unpaved Traffic Island': 8,
            'Unpaved Median Island': 9,
            'Hidden Median': 10,
            'Hidden Road': 11
        }

        self.mapping_names_colors = {
            1: ('Alley', 'green'),
            2: ('Paved Drive', 'purple'),
            3: ('Parking Lot', 'red'),
            4: ('Road', 'blue'),
            5: ('Intersection', 'cyan'),
            6: ('Paved Median Island', 'pink'),
            7: ('Paved Traffic Island', 'yellow'),
            8: ('Unpaved Traffic Island', 'brown'),
            9: ('Unpaved Median Island', 'grey'),
            10: ('Hidden Median', 'orange'),
            11: ('Hidden Road', 'white'),
        }


def show_image_from_np(photo=None, mask=None, figsize=(10, 10), ax=None):
    if (photo is not None or mask is not None) and ax is None:
        plt.figure(figsize=figsize)

    if photo is not None:
        if not np.issubdtype(photo.dtype, np.integer):
            photo = photo.astype('int32')
        if ax is not None:
            ax.imshow(photo)
        else:
            plt.imshow(photo)

    if mask is not None:
        if not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype('int32')
        unique_values = np.unique(mask)
        unique_values = unique_values[unique_values > 0]
        if not np.issubdtype(mask.dtype, np.floating):
            mask = mask.astype('float32')
        mask[np.where(mask == 0)] = np.nan
        color_map = ListedColormap([SurfaceMapping().mapping_names_colors[pos][1] for pos in range(1, 12)])
        legend_handles = [mpatches.Patch(color=SurfaceMapping().mapping_names_colors[value][1],
                                         label=SurfaceMapping().mapping_names_colors[value][0])
                          for value in unique_values]
        if ax is not None:
            ax.imshow(mask, alpha=0.5, cmap=color_map, vmin=1, vmax=11)
            ax.legend(handles=legend_handles, title='Pixel Values')
        else:
            plt.imshow(mask, alpha=0.5, cmap=color_map, vmin=1, vmax=11)
            plt.legend(handles=legend_handles, title='Pixel Values')
    if (photo is not None or mask is not None) and ax is None:
        plt.show()


def compare_predicted_mask(photo=None, true_mask=None, pred_mask=None, figsize=(20, 10)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    show_image_from_np(photo, true_mask, ax=axs[0])
    show_image_from_np(photo, pred_mask, ax=axs[1])
    plt.show()

def visualize_data(dataset_dir, show_masks=True, max_samples=5, shuffle=True, labels=None, only=False):
    photo_folder = os.path.join(dataset_dir, "photos")
    mask_folder = os.path.join(dataset_dir, "masks")

    file_list = os.listdir(photo_folder)

    if shuffle:
        random.shuffle(file_list)

    cnt = 0
    for file_name in file_list:
        if file_name.startswith("photo") and cnt < max_samples:
            mask_file_name = "mask_" + file_name[6:-4] + ".tif"

            photo_path = os.path.join(photo_folder, file_name)
            mask_path = os.path.join(mask_folder, mask_file_name)

            if show_masks:
                with rasterio.open(mask_path) as mask_ds:
                    mask = mask_ds.read(1)  # Assuming single band mask

                    # these conditions don't work as intended
                    if only and labels and any(np.setdiff1d(mask, labels)) and not all(np.isin(labels, mask)):
                        continue
                    elif not only and labels and any(np.isin(labels, mask)):  # this one especially
                        continue
            else:
                mask = None

            with rasterio.open(photo_path) as photo_ds:
                # photo = photo_ds.read([1,2,3])  # Assuming single band image
                photo = np.transpose(photo_ds.read([1, 2, 3]), (1, 2, 0))  # Load n layers

            show_image_from_np(photo, mask)
            cnt += 1
