import argparse
import itertools
import math
import unittest

import numpy as np
import torch
from scipy.special import eval_jacobi

from model_config import MODEL_DEFAULTS, add_model_arguments, architecture_options
from U_VixLSTM.UVixLSTM import (
    GlobalZernikeConv3d,
    MAX_ZERNIKE_INDEX,
    UVixLSTM,
    ansi_index_to_order,
    zernike_mode,
)


def legacy_mode(j, X, Y, Z, R):
    """The four modes exactly as the original ``_get_zernike_mode`` computed them."""
    if j == 0: return torch.ones_like(X)
    if j == 3: return Z
    if j == 4: return 2*R**2 - 1
    if j == 12: return 6.*R**4 - 6.*R**2 + 1.
    return torch.zeros_like(X)


def textbook_2d_mode(j, ky, kx):
    """Unnormalised OSA/ANSI Zernike mode from the Jacobi-polynomial form of R_n^m."""
    n, m = ansi_index_to_order(j)
    order = abs(m)
    rho, phi = np.hypot(kx, ky), np.arctan2(ky, kx)
    k = (n - order) // 2
    radial = (-1) ** k * rho ** order * eval_jacobi(k, order, 0, 1 - 2 * rho ** 2)
    if m == 0:
        return radial
    return radial * (np.cos(order * phi) if m > 0 else np.sin(order * phi))


def frequency_grid(shape=(8, 10, 12)):
    D, H, W = shape
    Z, Y, X = torch.meshgrid(torch.fft.fftfreq(D), torch.fft.fftfreq(H), torch.fft.rfftfreq(W), indexing="ij")
    return Z, Y, X


class ZernikeModeTests(unittest.TestCase):
    def test_original_modes_are_bit_identical(self):
        Z, Y, X = frequency_grid()
        R = torch.sqrt(X**2 + Y**2 + Z**2)
        for j in (0, 3, 4, 12):
            self.assertTrue(torch.equal(zernike_mode(j, Z, Y, X), legacy_mode(j, X, Y, Z, R)), f"j={j}")

    def test_index_to_order_follows_ansi(self):
        expected = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1),
                    (3, 3), (4, -4), (4, -2), (4, 0), (4, 2), (4, 4)]
        self.assertEqual([ansi_index_to_order(j) for j in range(15)], expected)
        self.assertEqual(ansi_index_to_order(24), (6, 0))
        self.assertEqual(ansi_index_to_order(MAX_ZERNIKE_INDEX), (8, -8))

    def test_lateral_plane_matches_textbook_2d_zernikes(self):
        rng = np.random.default_rng(0)
        kx, ky = rng.uniform(-0.5, 0.5, (2, 400))
        zero = torch.zeros(400)
        for j in range(MAX_ZERNIKE_INDEX + 1):
            if j == 3:  # the axial tilt replaces ANSI's oblique astigmatism
                continue
            ours = zernike_mode(j, zero, torch.tensor(ky, dtype=torch.float32), torch.tensor(kx, dtype=torch.float32))
            np.testing.assert_allclose(ours.numpy(), textbook_2d_mode(j, ky, kx), atol=2e-5, err_msg=f"j={j}")

    def test_tilts_are_exact_translations(self):
        Z, Y, X = frequency_grid()
        self.assertTrue(torch.equal(zernike_mode(1, Z, Y, X), Y))
        self.assertTrue(torch.equal(zernike_mode(2, Z, Y, X), X))
        self.assertTrue(torch.equal(zernike_mode(3, Z, Y, X), Z))

    def test_every_mode_is_finite_including_on_the_z_axis(self):
        Z, Y, X = frequency_grid()
        self.assertEqual(float(X[:, 0, 0].abs().max() + Y[:, 0, 0].abs().max()), 0.0)  # kx = ky = 0 exists
        for j in range(MAX_ZERNIKE_INDEX + 1):
            self.assertTrue(torch.isfinite(zernike_mode(j, Z, Y, X)).all(), f"j={j}")

    def test_invalid_indices_are_rejected(self):
        for bad in ([], [MAX_ZERNIKE_INDEX + 1], [-1], [3, 3], [1.5], [True], ["3"]):
            with self.assertRaises(ValueError, msg=str(bad)):
                GlobalZernikeConv3d(j_indices=bad)

    def test_every_index_up_to_the_maximum_trains(self):
        layer = GlobalZernikeConv3d(j_indices=list(range(MAX_ZERNIKE_INDEX + 1)), dropout_p=0.0)
        x = torch.randn(1, 1, 16, 16, 16)
        layer(x).square().sum().backward()
        inert = [j for j, grad in zip(layer.j_indices, layer.alphas.grad) if grad == 0]
        self.assertEqual(inert, [])

    def test_default_layer_is_unchanged(self):
        layer = GlobalZernikeConv3d()
        self.assertEqual(layer.j_indices, [3, 4, 12])
        self.assertEqual(tuple(layer.alphas.shape), (3,))


class ModelOptionTests(unittest.TestCase):
    def test_defaults_fill_in_for_older_configs(self):
        options = architecture_options({"optimizer": "sgd", "use_sgg_layer": False})
        self.assertEqual(options, {**MODEL_DEFAULTS, "use_sgg_layer": False})
        self.assertEqual(options["zernike_moments"], [3, 4, 12])

    def test_switches_must_be_real_booleans(self):
        for bad in ("false", 0, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                architecture_options({"use_cell_hint": bad})

    def test_moments_are_only_checked_when_the_layer_is_on(self):
        self.assertEqual(architecture_options({"zernike_enabled": False, "zernike_moments": []})["zernike_enabled"], False)
        with self.assertRaises(ValueError):
            architecture_options({"zernike_moments": []})
        with self.assertRaises(ValueError):
            architecture_options({"zernike_moments": [99]})

    def test_command_line_overrides(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args = parser.parse_args(["--no-use_cell_hint", "--use_sgg_layer", "--zernike_moments", "3", "4", "24"])
        self.assertEqual((args.use_cell_hint, args.use_sgg_layer, args.zernike_enabled), (False, True, None))
        self.assertEqual(args.zernike_moments, [3, 4, 24])

    def test_every_on_off_combination_builds_and_runs(self):
        x = torch.randn(1, 1, 64, 64, 64)
        hint = torch.full_like(x, -5.0)
        for sgg, cell_hint, zernike in itertools.product([True, False], repeat=3):
            cfg = {"use_sgg_layer": sgg, "use_cell_hint": cell_hint, "zernike_enabled": zernike,
                   "zernike_moments": [3, 4, 12, 24]}
            model = UVixLSTM(class_num=1, img_dim=64, out_channels=64, depth=1, **architecture_options(cfg)).eval()
            with torch.no_grad():
                plain, _ = model(x)
                self.assertEqual(tuple(plain.shape), (1, 1, 64, 64, 64))
                if cell_hint:
                    hinted, _ = model(x, hint)
                    self.assertEqual(tuple(hinted.shape), (1, 1, 64, 64, 64))
                else:
                    with self.assertRaises(ValueError):
                        model(x, hint)
            names = [name for name, _ in model.named_parameters()]
            label = f"sgg={sgg} hint={cell_hint} zernike={zernike}"
            self.assertEqual(any(n.startswith("cell_clue_layer") for n in names), cell_hint, label)
            self.assertEqual(any("msg_layer" in n for n in names), sgg, label)
            self.assertEqual(any("zernike.alphas" in n for n in names), zernike, label)
            if zernike:
                self.assertEqual(tuple(model.encoder.zernike.alphas.shape), (4,), label)


if __name__ == "__main__":
    unittest.main()
