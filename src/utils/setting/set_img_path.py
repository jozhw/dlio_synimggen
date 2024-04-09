import os


def set_img_path(
    path: str,
    save_dir: str,
) -> str:

    # set image path for polaris
    if save_dir == "polaris":
        image_path: str = os.path.join(os.path.expanduser("~"), "..", "..", path)
    else:
        image_path: str = os.path.expanduser("{}".format(path))

    return image_path
