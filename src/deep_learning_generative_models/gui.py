"""Tkinter GUI for exploring trained AE and VAE checkpoints."""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from torch import Tensor, nn

from deep_learning_generative_models.config import ArchitecturePreset, ExperimentConfig
from deep_learning_generative_models.data import build_fashion_mnist_loaders
from deep_learning_generative_models.device import DeviceInfo, get_device
from deep_learning_generative_models.generate import generate_images
from deep_learning_generative_models.models import (
    ModelKind,
    VAEForwardOutput,
    VariationalAutoencoder,
    get_model_description,
)
from deep_learning_generative_models.train import CHECKPOINT_FILENAME, load_model_checkpoint

PREVIEW_IMAGE_COUNT = 24
DISPLAY_SCALE = 6
LATENT_SLIDER_LIMIT = 8
LATENT_SLIDER_MIN = -3.0
LATENT_SLIDER_MAX = 3.0


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    model: nn.Module
    checkpoint: dict[str, object]
    config: ExperimentConfig


class ModelExplorerService:
    """Non-visual controller logic used by the Tkinter layer."""

    def __init__(self, device_info: DeviceInfo | None = None) -> None:
        self.device_info = device_info or get_device()
        self.loaded: LoadedCheckpoint | None = None
        self.test_images: list[Tensor] = []
        self.test_labels: list[int] = []
        self.encoded_latent: Tensor | None = None

    def architecture_description(
        self,
        model_type: ModelKind | str,
        architecture_preset: ArchitecturePreset | str,
    ) -> str:
        return get_model_description(model_type, architecture_preset)

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        expected_model_type: ModelKind | str,
        expected_preset: ArchitecturePreset | str,
        preview_count: int = PREVIEW_IMAGE_COUNT,
    ) -> LoadedCheckpoint:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model, checkpoint = load_model_checkpoint(path, map_location=self.device_info.device)
        config = ExperimentConfig(**checkpoint["config"])
        if config.model_type != expected_model_type:
            raise ValueError(
                f"Checkpoint model type is {config.model_type!r}, "
                f"not {expected_model_type!r}"
            )
        if config.architecture_preset != expected_preset:
            raise ValueError(
                f"Checkpoint preset is {config.architecture_preset!r}, "
                f"not {expected_preset!r}"
            )

        model.to(self.device_info.device)
        model.eval()
        self.loaded = LoadedCheckpoint(
            path=path,
            model=model,
            checkpoint=checkpoint,
            config=config,
        )
        self.encoded_latent = None
        self._load_test_examples(config, preview_count)
        return self.loaded

    def reconstruct_selected(self, index: int) -> tuple[Tensor, Tensor]:
        loaded = self._require_loaded()
        image = self._test_image(index)
        with torch.no_grad():
            batch = image.unsqueeze(0).to(self.device_info.device)
            reconstruction = model_reconstruction(loaded.model, batch).squeeze(0).cpu()
        return image, reconstruction

    def generate_random(self, count: int = 4, seed: int | None = None) -> Tensor:
        loaded = self._require_loaded()
        if not isinstance(loaded.model, VariationalAutoencoder):
            raise ValueError("Random generation requires a VAE checkpoint")
        return generate_images(
            model=loaded.model,
            count=count,
            device=self.device_info.device,
            seed=seed,
        )

    def vae_features_available(self) -> bool:
        return isinstance(self.loaded.model, VariationalAutoencoder) if self.loaded else False

    def encode_selected_image(self, index: int) -> Tensor:
        model = self._require_vae()
        image = self._test_image(index)
        with torch.no_grad():
            batch = image.unsqueeze(0).to(self.device_info.device)
            mu, _logvar = model.encode(batch)
            self.encoded_latent = mu.squeeze(0).detach().cpu()
        return self.encoded_latent.clone()

    def decode_latent(self, latent: Tensor) -> Tensor:
        model = self._require_vae()
        if latent.shape != (model.latent_dim,):
            raise ValueError(f"latent vector must have shape [{model.latent_dim}]")
        with torch.no_grad():
            batch = latent.unsqueeze(0).to(self.device_info.device)
            return model.decode(batch).squeeze(0).detach().cpu()

    def decode_modified_latent(self, base_latent: Tensor, values: dict[int, float]) -> Tensor:
        latent = modify_latent_vector(base_latent, values)
        return self.decode_latent(latent)

    def interpolate_selected_images(
        self,
        index_a: int,
        index_b: int,
        alpha: float,
    ) -> Tensor:
        model = self._require_vae()
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        image_a = self._test_image(index_a)
        image_b = self._test_image(index_b)

        with torch.no_grad():
            batch_a = image_a.unsqueeze(0).to(self.device_info.device)
            batch_b = image_b.unsqueeze(0).to(self.device_info.device)
            mu_a, _logvar_a = model.encode(batch_a)
            mu_b, _logvar_b = model.encode(batch_b)
            latent = interpolate_latent_vectors(
                mu_a.squeeze(0),
                mu_b.squeeze(0),
                alpha,
            )
            return model.decode(latent.unsqueeze(0)).squeeze(0).detach().cpu()

    def _load_test_examples(self, config: ExperimentConfig, preview_count: int) -> None:
        if preview_count <= 0:
            raise ValueError("preview_count must be positive")
        preview_config = replace(config, test_subset_size=preview_count)
        loaders = build_fashion_mnist_loaders(preview_config)
        dataset = loaders.test.dataset
        self.test_images = []
        self.test_labels = []
        for index in range(min(preview_count, len(dataset))):
            image, label = dataset[index]
            if not isinstance(image, Tensor):
                raise TypeError("Dataset image must be a tensor")
            self.test_images.append(image)
            self.test_labels.append(int(label))
        if not self.test_images:
            raise ValueError("No Fashion-MNIST test images were loaded")

    def _test_image(self, index: int) -> Tensor:
        if index < 0 or index >= len(self.test_images):
            raise IndexError("Selected image index is out of range")
        return self.test_images[index]

    def _require_loaded(self) -> LoadedCheckpoint:
        if self.loaded is None:
            raise ValueError("Load a compatible checkpoint first")
        return self.loaded

    def _require_vae(self) -> VariationalAutoencoder:
        loaded = self._require_loaded()
        if not isinstance(loaded.model, VariationalAutoencoder):
            raise ValueError("Latent-space exploration requires a VAE checkpoint")
        return loaded.model


def model_reconstruction(model: nn.Module, images: Tensor) -> Tensor:
    output = model(images)
    if isinstance(output, VAEForwardOutput):
        return output.reconstruction
    if isinstance(output, Tensor):
        return output
    raise TypeError("Model output must be a reconstruction tensor or VAEForwardOutput")


def interpolate_latent_vectors(start: Tensor, end: Tensor, alpha: float) -> Tensor:
    if start.shape != end.shape:
        raise ValueError("start and end latent tensors must have matching shapes")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    return (1.0 - alpha) * start + alpha * end


def modify_latent_vector(base_latent: Tensor, values: dict[int, float]) -> Tensor:
    if base_latent.ndim != 1:
        raise ValueError("base_latent must be one-dimensional")
    modified = base_latent.detach().clone()
    for index, value in values.items():
        if index < 0 or index >= modified.shape[0]:
            raise IndexError("latent dimension index is out of range")
        bounded = max(LATENT_SLIDER_MIN, min(LATENT_SLIDER_MAX, float(value)))
        modified[index] = bounded
    return modified


def tensor_to_tk_color_rows(image: Tensor, scale: int = DISPLAY_SCALE) -> list[str]:
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("image must have shape [1, height, width]")
    if scale <= 0:
        raise ValueError("scale must be positive")

    pixels = image.detach().cpu().clamp(0.0, 1.0).mul(255).to(torch.uint8).squeeze(0)
    if scale != 1:
        pixels = pixels.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)

    rows: list[str] = []
    for row in pixels.tolist():
        colors = [f"#{value:02x}{value:02x}{value:02x}" for value in row]
        rows.append("{" + " ".join(colors) + "}")
    return rows


def image_grid_tensor(images: Tensor, columns: int | None = None) -> Tensor:
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("images must have shape [count, 1, 28, 28]")
    if images.shape[0] == 0:
        raise ValueError("At least one image is required")

    count = images.shape[0]
    columns = columns or math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    grid = torch.zeros(1, rows * 28, columns * 28)
    for index, image in enumerate(images.detach().cpu()):
        row = index // columns
        column = index % columns
        grid[:, row * 28 : (row + 1) * 28, column * 28 : (column + 1) * 28] = image
    return grid


class ModelExplorerApp:
    def __init__(self, root: tk.Tk, service: ModelExplorerService | None = None) -> None:
        self.root = root
        self.service = service or ModelExplorerService()
        self.model_type_var = tk.StringVar(value="ae")
        self.preset_var = tk.StringVar(value="small")
        self.checkpoint_var = tk.StringVar(value="")
        self.image_index_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value=self.service.device_info.description)
        self.description_var = tk.StringVar(value="")
        self.image_b_index_var = tk.IntVar(value=1)
        self.interpolation_alpha_var = tk.DoubleVar(value=0.0)
        self.latent_slider_vars: list[tk.DoubleVar] = []
        self.latent_sliders: list[ttk.Scale] = []
        self.latent_base: Tensor | None = None
        self._image_refs: list[tk.PhotoImage] = []

        self.root.title("AE/VAE Model Explorer")
        self._build_ui()
        self._update_description()
        self._update_vae_control_state()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="Model").grid(row=0, column=0, padx=(0, 4))
        model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_type_var,
            values=("ae", "vae"),
            width=8,
            state="readonly",
        )
        model_combo.grid(row=0, column=1, padx=(0, 12))
        model_combo.bind("<<ComboboxSelected>>", self._on_model_selection_changed)

        ttk.Label(controls, text="Preset").grid(row=0, column=2, padx=(0, 4))
        preset_combo = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=("small", "medium", "deep"),
            width=8,
            state="readonly",
        )
        preset_combo.grid(row=0, column=3, padx=(0, 12))
        preset_combo.bind("<<ComboboxSelected>>", self._on_model_selection_changed)

        ttk.Button(controls, text="Load Checkpoint", command=self._choose_checkpoint).grid(
            row=0,
            column=4,
            padx=(0, 8),
        )
        ttk.Label(controls, textvariable=self.checkpoint_var).grid(
            row=0,
            column=5,
            sticky="ew",
        )

        ttk.Label(main, textvariable=self.description_var, wraplength=780).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 8),
        )

        browser = ttk.Frame(main)
        browser.grid(row=2, column=0, sticky="ew")
        ttk.Label(browser, text="Test image").grid(row=0, column=0, padx=(0, 4))
        self.image_combo = ttk.Combobox(
            browser,
            textvariable=self.image_index_var,
            values=(),
            width=24,
            state="readonly",
        )
        self.image_combo.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(browser, text="Reconstruct", command=self._reconstruct).grid(
            row=0,
            column=2,
            padx=(0, 8),
        )
        self.generate_button = ttk.Button(
            browser,
            text="Generate Random",
            command=self._generate_random,
        )
        self.generate_button.grid(row=0, column=3)

        latent = ttk.LabelFrame(main, text="VAE latent controls", padding=8)
        latent.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        ttk.Label(latent, text="Image B").grid(row=0, column=0, padx=(0, 4))
        self.image_b_combo = ttk.Combobox(
            latent,
            textvariable=self.image_b_index_var,
            values=(),
            width=24,
            state="disabled",
        )
        self.image_b_combo.grid(row=0, column=1, padx=(0, 8))

        ttk.Label(latent, text="A to B").grid(row=0, column=2, padx=(0, 4))
        self.interpolation_slider = ttk.Scale(
            latent,
            from_=0.0,
            to=1.0,
            variable=self.interpolation_alpha_var,
            command=self._interpolate_from_slider,
        )
        self.interpolation_slider.grid(row=0, column=3, sticky="ew", padx=(0, 8))
        latent.columnconfigure(3, weight=1)

        self.encode_button = ttk.Button(
            latent,
            text="Encode Selected",
            command=self._encode_selected_for_latent_sliders,
        )
        self.encode_button.grid(row=0, column=4)

        self.latent_sliders_frame = ttk.Frame(latent)
        self.latent_sliders_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(8, 0))

        display = ttk.Frame(main)
        display.grid(row=4, column=0, sticky="nsew", pady=(12, 8))
        main.rowconfigure(4, weight=1)
        self.original_label = self._image_panel(display, "Original", 0)
        self.reconstruction_label = self._image_panel(display, "Reconstruction", 1)
        self.generated_label = self._image_panel(display, "Generated", 2)

        ttk.Label(main, textvariable=self.status_var, wraplength=780).grid(
            row=5,
            column=0,
            sticky="ew",
        )

    def _image_panel(self, parent: ttk.Frame, title: str, column: int) -> ttk.Label:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=0, column=column, padx=6, sticky="n")
        label = ttk.Label(frame)
        label.grid(row=0, column=0)
        return label

    def _on_model_selection_changed(self, _event: object | None = None) -> None:
        self._update_description()
        self._update_vae_control_state()

    def _choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=(("PyTorch checkpoint", "*.pt"), ("All files", "*.*")),
        )
        if path:
            self.load_checkpoint(Path(path))

    def load_checkpoint(self, path: Path) -> None:
        try:
            loaded = self.service.load_checkpoint(
                checkpoint_path=path,
                expected_model_type=self.model_type_var.get(),
                expected_preset=self.preset_var.get(),
            )
        except Exception as error:  # noqa: BLE001 - recoverable GUI boundary
            self._show_error(str(error))
            return

        self.checkpoint_var.set(str(loaded.path))
        self._set_image_choices()
        self.status_var.set(f"Loaded {loaded.config.model_type.upper()} checkpoint.")
        self._reset_latent_controls()
        self._update_vae_control_state()

    def _set_image_choices(self) -> None:
        choices = [
            f"{index}: label {label}"
            for index, label in enumerate(self.service.test_labels)
        ]
        self.image_combo.configure(values=choices)
        self.image_b_combo.configure(values=choices)
        if choices:
            self.image_combo.current(0)
            self.image_b_combo.current(1 if len(choices) > 1 else 0)

    def _reconstruct(self) -> None:
        try:
            original, reconstruction = self.service.reconstruct_selected(
                self._selected_image_index()
            )
        except Exception as error:  # noqa: BLE001 - recoverable GUI boundary
            self._show_error(str(error))
            return

        self._set_panel_image(self.original_label, original)
        self._set_panel_image(self.reconstruction_label, reconstruction)
        self.status_var.set("Reconstruction complete.")

    def _generate_random(self) -> None:
        try:
            generated = self.service.generate_random(count=4)
        except Exception as error:  # noqa: BLE001 - recoverable GUI boundary
            self._show_error(str(error))
            return

        self._set_panel_image(self.generated_label, image_grid_tensor(generated, columns=2))
        self.status_var.set("Random VAE generation complete.")

    def _interpolate_from_slider(self, _value: object | None = None) -> None:
        if not self.service.vae_features_available():
            return
        try:
            image = self.service.interpolate_selected_images(
                self._selected_image_index(),
                self._selected_image_b_index(),
                self.interpolation_alpha_var.get(),
            )
        except Exception as error:  # noqa: BLE001 - recoverable GUI boundary
            self._show_error(str(error))
            return
        self._set_panel_image(self.generated_label, image)
        self.status_var.set("Latent interpolation updated.")

    def _encode_selected_for_latent_sliders(self) -> None:
        try:
            self.latent_base = self.service.encode_selected_image(
                self._selected_image_index()
            )
        except Exception as error:  # noqa: BLE001 - recoverable GUI boundary
            self._show_error(str(error))
            return
        self._build_latent_sliders(self.latent_base)
        decoded = self.service.decode_latent(self.latent_base)
        self._set_panel_image(self.generated_label, decoded)
        self.status_var.set("Selected image encoded for latent editing.")

    def _build_latent_sliders(self, latent: Tensor) -> None:
        for child in self.latent_sliders_frame.winfo_children():
            child.destroy()
        self.latent_slider_vars = []
        self.latent_sliders = []
        count = min(LATENT_SLIDER_LIMIT, latent.shape[0])
        for index in range(count):
            ttk.Label(self.latent_sliders_frame, text=f"z{index}").grid(
                row=index // 4,
                column=(index % 4) * 2,
                padx=(0, 4),
                sticky="e",
            )
            variable = tk.DoubleVar(value=float(latent[index].clamp(-3.0, 3.0)))
            slider = ttk.Scale(
                self.latent_sliders_frame,
                from_=LATENT_SLIDER_MIN,
                to=LATENT_SLIDER_MAX,
                variable=variable,
                command=self._decode_latent_slider_values,
            )
            slider.grid(
                row=index // 4,
                column=(index % 4) * 2 + 1,
                sticky="ew",
                padx=(0, 10),
            )
            self.latent_sliders_frame.columnconfigure((index % 4) * 2 + 1, weight=1)
            self.latent_slider_vars.append(variable)
            self.latent_sliders.append(slider)

    def _decode_latent_slider_values(self, _value: object | None = None) -> None:
        if self.latent_base is None or not self.service.vae_features_available():
            return
        values = {
            index: variable.get()
            for index, variable in enumerate(self.latent_slider_vars)
        }
        try:
            decoded = self.service.decode_modified_latent(self.latent_base, values)
        except Exception as error:  # noqa: BLE001 - recoverable GUI boundary
            self._show_error(str(error))
            return
        self._set_panel_image(self.generated_label, decoded)
        self.status_var.set("Latent vector decoded.")

    def _selected_image_index(self) -> int:
        value = self.image_combo.get()
        if not value:
            raise ValueError("Select a Fashion-MNIST test image first")
        return int(value.split(":", maxsplit=1)[0])

    def _selected_image_b_index(self) -> int:
        value = self.image_b_combo.get()
        if not value:
            raise ValueError("Select a second Fashion-MNIST test image first")
        return int(value.split(":", maxsplit=1)[0])

    def _set_panel_image(self, label: ttk.Label, image: Tensor) -> None:
        rows = tensor_to_tk_color_rows(image)
        photo = tk.PhotoImage(width=len(rows[0].split()), height=len(rows))
        photo.put(" ".join(rows), to=(0, 0))
        label.configure(image=photo)
        self._image_refs.append(photo)

    def _update_description(self) -> None:
        try:
            description = self.service.architecture_description(
                self.model_type_var.get(),
                self.preset_var.get(),
            )
        except ValueError as error:
            description = str(error)
        self.description_var.set(description)

    def _update_vae_control_state(self) -> None:
        loaded_vae = self.service.vae_features_available()
        selected_vae = self.model_type_var.get() == "vae"
        state = "normal" if loaded_vae and selected_vae else "disabled"
        self.generate_button.configure(state=state)
        self.image_b_combo.configure(state="readonly" if state == "normal" else "disabled")
        self.interpolation_slider.configure(state=state)
        self.encode_button.configure(state=state)
        for slider in self.latent_sliders:
            slider.configure(state=state)

    def _reset_latent_controls(self) -> None:
        self.latent_base = None
        self.interpolation_alpha_var.set(0.0)
        for child in self.latent_sliders_frame.winfo_children():
            child.destroy()
        self.latent_slider_vars = []
        self.latent_sliders = []

    def _show_error(self, message: str) -> None:
        self.status_var.set(message)
        messagebox.showerror("Model Explorer", message)


def run_gui() -> None:
    root = tk.Tk()
    ModelExplorerApp(root)
    root.mainloop()


def smoke_test_gui(
    ae_checkpoint: Path | str,
    vae_checkpoint: Path | str,
) -> None:
    root = tk.Tk()
    root.withdraw()
    app = ModelExplorerApp(root)
    app.model_type_var.set("ae")
    app.preset_var.set("small")
    app._on_model_selection_changed()
    app.load_checkpoint(Path(ae_checkpoint))
    app._reconstruct()
    app.model_type_var.set("vae")
    app.preset_var.set("small")
    app._on_model_selection_changed()
    app.load_checkpoint(Path(vae_checkpoint))
    app._reconstruct()
    app._generate_random()
    app._interpolate_from_slider()
    app._encode_selected_for_latent_sliders()
    if app.latent_slider_vars:
        app.latent_slider_vars[0].set(0.5)
        app._decode_latent_slider_values()
    root.update_idletasks()
    root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the Tkinter AE/VAE model explorer."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a non-interactive GUI smoke check and close.",
    )
    parser.add_argument("--ae-checkpoint", help=f"Path to an AE {CHECKPOINT_FILENAME}.")
    parser.add_argument("--vae-checkpoint", help=f"Path to a VAE {CHECKPOINT_FILENAME}.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        if not args.ae_checkpoint or not args.vae_checkpoint:
            raise SystemExit("--smoke-test requires --ae-checkpoint and --vae-checkpoint")
        smoke_test_gui(args.ae_checkpoint, args.vae_checkpoint)
        print("GUI smoke test complete.")
        return
    run_gui()


if __name__ == "__main__":
    main()
