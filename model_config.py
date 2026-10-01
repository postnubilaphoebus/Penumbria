"""YAML options of the model architecture: defaults, validation and construction.

The four options live in the ``model`` section of a dataset config and are
independent of each other, so every on/off combination is valid:

    use_sgg_layer:   true | false   multi-scale graph (SGG) branch
    use_cell_hint:   true | false   cell-hint prompt branch
    zernike_enabled: true | false   Zernike phase layer
    zernike_moments: [3, 4, 12]     ANSI/OSA indices j in 0..36 (ignored when the layer is off)
"""
import argparse

from U_VixLSTM.UVixLSTM import UVixLSTM, DEFAULT_ZERNIKE_MOMENTS, validate_zernike_indices

MODEL_SWITCHES = ("use_sgg_layer", "use_cell_hint", "zernike_enabled")

MODEL_DEFAULTS = {
    "use_sgg_layer": True,
    "use_cell_hint": True,
    "zernike_enabled": True,
    "zernike_moments": list(DEFAULT_ZERNIKE_MOMENTS),
}


def with_model_defaults(model_cfg):
    """Add the architecture options that an older YAML does not mention."""
    return {**MODEL_DEFAULTS, **model_cfg}


def architecture_options(model_cfg):
    """The validated architecture options of a ``model`` config section."""
    options = with_model_defaults(model_cfg)
    for key in MODEL_SWITCHES:
        if type(options[key]) is not bool:
            raise ValueError(f"model.{key} must be true or false, got {options[key]!r}")
    if options["zernike_enabled"]:
        options["zernike_moments"] = validate_zernike_indices(options["zernike_moments"])
    return {key: options[key] for key in (*MODEL_SWITCHES, "zernike_moments")}


def build_model(model_cfg, img_dim):
    """UVixLSTM for the given ``model`` config section and cubic training patch size."""
    return UVixLSTM(class_num=1, img_dim=img_dim, out_channels=64, depth=12, dim=256,
                    **architecture_options(model_cfg))


def add_model_arguments(parser):
    """Command-line overrides for the four architecture options."""
    for key in MODEL_SWITCHES:
        parser.add_argument(f'--{key}', action=argparse.BooleanOptionalAction, default=None,
                            help=f'Override model.{key} (use --{key} or --no-{key})')
    parser.add_argument('--zernike_moments', type=int, nargs='+',
                        help='ANSI/OSA Zernike indices j (0-36) used by the Zernike layer, e.g. 3 4 12')
