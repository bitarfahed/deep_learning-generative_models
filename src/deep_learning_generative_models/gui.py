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


def model_reconstruction(model: nn.Module, images: Tensor) -> Tensor:
    output = model(images)
    if isinstance(output, VAEForwardOutput):
        return output.reconstruction
    if isinstance(output, Tensor):
        return output
    raise TypeError("Model output must be a reconstruction tensor or VAEForwardOutput")


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
        self._image_refs: list[tk.PhotoImage] = []

        self.root.title("AE/VAE Model Explorer")
        self._build_ui()
        self._update_description()
        self._update_generation_state()

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

        display = ttk.Frame(main)
        display.grid(row=3, column=0, sticky="nsew", pady=(12, 8))
        main.rowconfigure(3, weight=1)
        self.original_label = self._image_panel(display, "Original", 0)
        self.reconstruction_label = self._image_panel(display, "Reconstruction", 1)
        self.generated_label = self._image_panel(display, "Generated", 2)

        ttk.Label(main, textvariable=self.status_var, wraplength=780).grid(
            row=4,
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
        self._update_generation_state()

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
        self._update_generation_state()

    def _set_image_choices(self) -> None:
        choices = [
            f"{index}: label {label}"
            for index, label in enumerate(self.service.test_labels)
        ]
        self.image_combo.configure(values=choices)
        if choices:
            self.image_combo.current(0)

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

    def _selected_image_index(self) -> int:
        value = self.image_combo.get()
        if not value:
            raise ValueError("Select a Fashion-MNIST test image first")
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

    def _update_generation_state(self) -> None:
        state = "normal" if self.model_type_var.get() == "vae" else "disabled"
        self.generate_button.configure(state=state)

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
