# %%
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from pathlib import Path
from importlib.machinery import SourceFileLoader
from cellmap_models import download_url_to_file

default_params = {
    "in_channels": 1,
    "downsample_factors": [(2,) * 3, (3,) * 3, (3,) * 3],
    "kernel_size_down": None,
    "kernel_size_up": None,
    "feature_widths_down": [12, 12 * 6, 12 * 6**2, 12 * 6**3],
    "feature_widths_up": [12 * 6, 12 * 6, 12 * 6**2, 12 * 6**3],
    "activation": "ReLU",
    "constant_upsample": True,
    "padding": "valid",
    "activation_on_upsample": False,
}


def get_param_dict(model_params):
    param_dict = default_params.copy()
    for field in default_params.keys():
        if hasattr(model_params, field):
            param_dict[field] = getattr(model_params, field)
    return param_dict


def load_model(checkpoint_name: str) -> torch.nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_name (str): Name of the checkpoint file.
    """
    from . import models_dict, models_list, model_names  # avoid circular import

    # Make sure the checkpoint exists
    if (
        checkpoint_name not in models_dict
        and Path(checkpoint_name).with_suffix(".pth") not in models_list
    ):
        if checkpoint_name in model_names:
            checkpoint_path = Path(
                Path(__file__).parent / Path(checkpoint_name) / "model.py"
            )
            no_weights = True
        else:
            raise ValueError(f"Model {checkpoint_name} not found")
    else:
        checkpoint_path = Path(
            Path(__file__).parent / Path(checkpoint_name)
        ).with_suffix(".pth")
        if not checkpoint_path.exists():
            url = models_dict[checkpoint_name]
            print(f"Downloading {checkpoint_name} from {url}")
            download_url_to_file(url, checkpoint_path)
        no_weights = False

    model_params = SourceFileLoader(
        "model", str(Path(checkpoint_path).parent / "model.py")
    ).load_module()

    model = Architecture(model_params)

    if no_weights:
        print(f"Not loading weights for {checkpoint_name}.")
        return model

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = deepcopy(checkpoint)
    for key in checkpoint["model"].keys():
        if "chain" in key:
            new_checkpoint["model"].pop(key)
            continue
        new_key = key.replace("architecture.", "")
        new_checkpoint["model"][new_key] = new_checkpoint["model"].pop(key)
    model.load_state_dict(new_checkpoint["model"])
    model.eval()

    return model


class Architecture(torch.nn.Module):
    def __init__(self, model_params):
        super().__init__()
        model_param_dict = get_param_dict(model_params)
        unet = CNNectomeUNetModule(**model_param_dict)

        layers = [unet]
        if hasattr(model_params, "upsample_factor"):
            up = Upsample(
                model_params.upsample_factor,
                mode=(
                    "nearest"
                    if model_param_dict["constant_upsample"]
                    else "transposed_conv"
                ),
                in_channels=model_param_dict["feature_widths_up"][0],
                out_channels=model_param_dict["feature_widths_up"][0],
                activation=(
                    model_param_dict["activation"]
                    if model_param_dict["activation_on_upsample"]
                    else None
                ),
            )
            layers.append(up)
        layers.append(
            ConvPass(
                model_param_dict["feature_widths_up"][0],
                model_params.final_feature_width,
                model_params.final_kernel_size,
                activation=model_param_dict["activation"],
                padding=model_param_dict["padding"],
            )
        )
        prediction_head = torch.nn.Conv3d(
            model_params.final_feature_width,
            model_params.classes_out,
            kernel_size=(1,) * 3,
        )
        model = torch.nn.Sequential(*layers)
        self.unet = model
        self.prediction_head = prediction_head
        for k, v in model_params.__dict__.items():
            setattr(self, k, v)
        for k, v in model_param_dict.items():
            setattr(self, k, v)
        self.compute_minimal_shapes()

    def forward(self, x):
        return self.prediction_head(self.unet(x))

    def compute_minimal_shapes(self):
        """
        Computes the minimal input shape, shape at the bootleneck and output shape as well as suitable step sizes
        (additional context) for the given U-Net configuration. This is computed for U-Nets with `valid` padding as
        well as for U-Nets with `same` padding. For `same` padding U-Nets these requirements are not strict, but
        represent the minimum shape for which voxels that are seeing a full field of view are contained in the output
        and thus making it easy to switch to a `valid` padding U-Net for inference

        Returns:
            A 4-element tuple containing, respectively, the minimum input shape and valid step size, the corresponding
            minimum output shape and minimum bottleneck shape, i.e. shape after last downsampling.
        """

        # valid step (meaning what values can be added on top of the minimum shape to also produce a U-Net with valid
        # shapes
        step = np.prod(self.downsample_factors, axis=0)

        # PART 1: calculate the minimum shape of the feature map after convolutions in the bottom layer ("bottom
        # right") such that a feature map size of 1 can be guaranteed throughout the upsampling paths

        # initialize with a minimum shape of 1 (representing the size after convolutions in each level)
        min_bottom_right = [(1.0, 1.0, 1.0)] * (len(self.downsample_factors) + 1)

        # propagate those minimal shapes back through the network to calculate the corresponding minimal shapes on the
        # "bottom right"

        # iterate over levels of unet
        for lv in range(len(self.downsample_factors)):
            kernels = np.copy(self.kernel_sizes_up[lv])

            # padding added by convolution kernels on current level (upsampling path)
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )

            assert np.all(total_pad % 2 == 0), (
                "Kernels {kernels:} on level {lv:} of U-Net (upsampling path) not compatible with enforcing an "
                "even context".format(kernels=kernels, lv=lv)
            )

            # for translational equivariance U-Net includes cropping to the stride of the downsampling factors
            # rounding up the padding to the closest multiple of what is the crop factor because the result of the
            # upsampling will be a multiple of the crop factor, and crop_to_factor makes it such that after the
            # operations on this level the feature map will also be a multiple of the crop factor, i.e. the
            # total_pad needs to be a multiple of the crop factor as well

            total_pad = np.ceil(
                total_pad / np.prod(self.downsample_factors[lv:], axis=0, dtype=float)
            ) * np.prod(self.downsample_factors[lv:], axis=0)

            # when even context are enforced the padding needs to be even so trans_equivariant will crop +1
            # factors if the otherwise resulting padding is odd
            total_pad += (total_pad % 2) * np.prod(self.downsample_factors[lv:], axis=0)

            for l in range(lv + 1):
                min_bottom_right[l] += total_pad  # add the padding added by convolution
                min_bottom_right[l] /= self.downsample_factors[
                    lv
                ]  # divide by downsampling factor of current level

        # round up the fractions potentially introduced by downsampling factor division
        min_bottom_right = np.ceil(min_bottom_right)

        # take the max across levels (i.e. we find the level that required the most context)
        min_bottom_right = np.max(min_bottom_right, axis=0)

        # PART 2: calculate the minimum input shape by propagating from the "bottom right" to the input of the U-Net
        min_input_shape = np.copy(min_bottom_right)

        for lv in range(len(self.kernel_sizes_down))[
            ::-1
        ]:  # go backwards through downsampling path

            if lv != len(self.kernel_sizes_down) - 1:  # unless bottom layer
                min_input_shape *= self.downsample_factors[
                    lv
                ]  # calculate shape before downsampling

            # calculate shape before convolutions on current level
            kernels = np.copy(self.kernel_sizes_down[lv])
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            assert np.all(total_pad % 2 == 0), (
                "Kernels {kernels:} on level {lv:} of U-Net (downsampling path) not compatible with enforcing an "
                "even context".format(kernels=kernels, lv=lv)
            )

            min_input_shape += total_pad

        # PART 3: calculate the minimum output shape by propagating from the "bottom right" to the output of the U-Net
        min_output_shape = np.copy(min_bottom_right)
        for lv in range(len(self.downsample_factors))[
            ::-1
        ]:  # go through upsampling path
            min_output_shape *= self.downsample_factors[
                lv
            ]  # calculate shape after upsampling

            # calculate shape after convolutions on current level
            kernels = np.copy(self.kernel_sizes_up[lv])
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )

            # same rational for translational equivariance as above in PART 1
            total_pad = np.ceil(
                total_pad / np.prod(self.downsample_factors[lv:], axis=0, dtype=float)
            ) * np.prod(self.downsample_factors[lv:], axis=0)
            min_output_shape -= total_pad

        self.min_input_shape = min_input_shape
        self.min_output_shape = min_output_shape
        self.input_size_step = step


class CNNectomeUNetModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        feature_widths_down=None,
        feature_widths_up=None,
        activation="ReLU",
        num_heads=1,
        constant_upsample=False,
        padding="valid",
        activation_on_upsample=False,
    ):
        """Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        This U-Net performs only "valid" convolutions, i.e., sizes of the
        feature maps decrease after each convolution. It will perfrom 4D
        convolutions as long as ``length`` is greater than 1. As soon as
        ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            in_channels:

                The number of input channels.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            feature_widths_down (optional):

                List of integers to determine the number of feature maps in the
                different levels of the build on the left side.

            feature_widths_up (optional):

                List of integers to determine the number of feature maps in the
                different levels of the build on the right side.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            padding (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).

            activation_on_upsample:

                Whether or not to add an activation after the upsample operation.
        """

        super().__init__()

        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels

        self.dims = len(downsample_factors[0])

        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3,) * self.dims, (3,) * self.dims]] * self.num_levels
        self.kernel_size_down = kernel_size_down
        if kernel_size_up is None:
            kernel_size_up = [[(3,) * self.dims, (3,) * self.dims]] * (
                self.num_levels - 1
            )
        self.kernel_size_up = kernel_size_up
        if feature_widths_down is None:
            feature_widths_down = [12 * 6**i for i in range(self.num_levels)]
        self.feature_widths_down = feature_widths_down
        if feature_widths_up is None:
            feature_widths_up = [12 * 6**i for i in range(self.num_levels)]
        self.feature_widths_up = feature_widths_up

        self.out_channels = feature_widths_up[0]

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(f * ff for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList(
            [
                ConvPass(
                    (in_channels if level == 0 else feature_widths_down[level - 1]),
                    feature_widths_down[level],
                    kernel_size_down[level],
                    activation=activation,
                    padding=padding,
                )
                for level in range(self.num_levels)
            ]
        )
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList(
            [
                Downsample(downsample_factors[level])
                for level in range(self.num_levels - 1)
            ]
        )

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Upsample(
                            downsample_factors[level],
                            mode="nearest" if constant_upsample else "transposed_conv",
                            in_channels=feature_widths_up[level + 1],
                            out_channels=feature_widths_up[level],
                            crop_factor=crop_factors[level],
                            next_conv_kernel_sizes=kernel_size_up[level],
                            activation=activation if activation_on_upsample else None,
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )
        #  if num_fmaps_out is None or level != self.num_levels-1 else num_fmaps_out

        # right convolutional passes
        self.r_conv = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvPass(
                            (
                                feature_widths_up[level + 1]
                                + feature_widths_down[level]
                                if level == 0
                                else feature_widths_up[level]
                                + feature_widths_down[level]
                            ),
                            feature_widths_up[level],
                            kernel_size_up[level],
                            activation=activation,
                            padding=padding,
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

    def rec_forward(self, level, f_in):
        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:
            fs_out = [f_left] * self.num_heads

        else:
            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            # up, concat, and crop
            fs_right = [
                self.r_up[h][i](gs_out[h], f_left) for h in range(self.num_heads)
            ]

            # convolve
            fs_out = [self.r_conv[h][i](fs_right[h]) for h in range(self.num_heads)]

        return fs_out

    def forward(self, x):
        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        return y


class ConvPass(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_sizes, activation, padding="valid"
    ):
        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)

        layers = []

        for kernel_size in kernel_sizes:
            self.dims = len(kernel_size)

            conv = {
                2: torch.nn.Conv2d,
                3: torch.nn.Conv3d,
            }[self.dims]

            if padding == "same":
                pad = tuple(k // 2 for k in kernel_size)
            else:
                pad = 0

            try:
                layers.append(conv(in_channels, out_channels, kernel_size, padding=pad))
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            in_channels = out_channels

            if activation is not None:
                layers.append(activation())

        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_pass(x)


class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor):
        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d,  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(downsample_factor, stride=downsample_factor)

    def forward(self, x):
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.downsample_factor, self.dims - d)
                )

        return self.down(x)


class Upsample(torch.nn.Module):
    def __init__(
        self,
        scale_factor,
        mode="transposed_conv",
        in_channels=None,
        out_channels=None,
        crop_factor=None,
        next_conv_kernel_sizes=None,
        activation=None,
    ):
        super(Upsample, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)
        assert (crop_factor is None) == (
            next_conv_kernel_sizes is None
        ), "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes

        self.dims = len(scale_factor)

        layers = []

        if mode == "transposed_conv":
            up = {2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}[self.dims]

            layers.append(
                up(
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                )
            )

        else:
            layers.append(torch.nn.Upsample(scale_factor=scale_factor, mode=mode))
            conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[self.dims]
            layers.append(
                conv(
                    in_channels,
                    out_channels,
                    kernel_size=(1,) * self.dims,
                    stride=(1,) * self.dims,
                ),
            )
        if activation is not None:
            layers.append(activation())

        if len(layers) > 1:
            self.up = torch.nn.Sequential(*layers)
        else:
            self.up = layers[0]

    def crop_to_factor(self, x, factor, kernel_sizes):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        """

        shape = x.size()
        spatial_shape = shape[-self.dims :]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes) for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n * f + c for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:
            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with factor %s and following "
                "convolutions %s" % (shape, factor, kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, g_out, f_left=None):
        g_up = self.up(g_out)

        if self.next_conv_kernel_sizes is not None:
            g_cropped = self.crop_to_factor(
                g_up, self.crop_factor, self.next_conv_kernel_sizes
            )
        else:
            g_cropped = g_up

        if f_left is not None:
            f_cropped = self.crop(f_left, g_cropped.size()[-self.dims :])

            return torch.cat([f_cropped, g_cropped], dim=1)
        else:
            return g_cropped


# %%
