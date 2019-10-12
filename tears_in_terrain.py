import io
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.animation import FFMpegFileWriter
import matplotlib.patches as patches
import scipy.stats as stats

BACKGROUND_COLOUR = "#000000FF"
FRAME_RATE = 24
LEFT_ALIGN = 0.08


@dataclass
class Scene:
    """
    Base class for animated scene layers
    """

    start_frame: int
    end_frame: int
    zorder: float
    render_frame: Generator[Image.Image, None, None]


def convert_plot_to_image(figure: figure.Figure) -> Image.Image:
    """
    Converts the specified Matplotlib Figure into a PIL Image
    :param figure: Figure to convert
    :return: PIL Image
    """
    buf = io.BytesIO()
    figure.savefig(buf, format="png", facecolor="None")
    buf.seek(0)
    im = Image.open(buf)
    return im


def draw_eye(
    axes_dims: List[float], persist_frames: int, fade_out_frames: int
) -> Generator[Image.Image, None, None]:
    interval_count = 361
    angle = np.linspace(0, np.pi * 2.0, interval_count)
    radius = np.array([num % 2 for num in range(0, interval_count)]) * 2.5 + 1.5
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    iris = np.vstack([x.reshape(1, -1), y.reshape(1, -1)])
    intervals = np.linspace(-7.05, 7.05, interval_count)
    positive_curve = 0.075 * intervals ** 2 - 3.75
    negative_curve = -0.075 * (intervals ** 2) + 3.75
    im: Image.Image = Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8))
    figure = plt.figure(figsize=(19.2, 10.8))

    for i in range(1, interval_count + 3, 3):
        figure.clear()

        # Draw Iris
        ax = figure.add_axes(axes_dims)
        ax.fill_between(
            intervals[interval_count - i :],
            positive_curve[interval_count - i :],
            negative_curve[interval_count - i :],
            color="white",
            zorder=1,
        )
        ax.plot(iris[0, 0:i], iris[1, 0:i], linewidth=5, color="blue", zorder=3)
        ax.fill_between(
            intervals,
            np.ones(interval_count) * 5,
            negative_curve,
            color="black",
            alpha=1.0,
            zorder=4,
        )
        ax.fill_between(
            intervals,
            -np.ones(interval_count) * 5,
            positive_curve,
            color="black",
            alpha=1.0,
            zorder=4,
        )
        ax.set_xlim(-9.6, 9.6)
        ax.set_ylim(-4.32, 4.32)
        ax.axis("off")
        patch = patches.Circle((0, 0), radius=4.02, color="black", zorder=2)
        ax.add_patch(patch)

        im = convert_plot_to_image(figure)
        yield im

    # Keep the image for this many frames
    for i in range(persist_frames):
        yield im

    # Fade out the image over this many frames
    fade_out_alpha = np.power(np.linspace(1, 0, fade_out_frames), 2)
    for alpha in fade_out_alpha:
        pixels = np.array(im)
        alpha_layer = pixels[:, :, 3]
        alpha_layer[alpha_layer > 0] = int(255 * alpha)
        yield Image.fromarray(pixels)

    # Stay black for the remainder
    black_screen = np.array(im)
    black_screen[:, :, :] = 0
    im = Image.fromarray(black_screen)
    while True:
        yield im


def draw_text(
    sentence: str,
    text_pos_list: List[int],
    alpha_transitions: int,
    persist_frames: int,
    fade_out_frames: int,
) -> Generator[Image.Image, None, None]:
    """
    Render the next frame
    :return: The next render as a PIL Image
    """
    im: Image.Image = Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8))
    figure = plt.figure(figsize=(19.2, 10.8))
    alpha_array = np.power(np.linspace(0, 1, alpha_transitions), 2)
    for idx, text_pos in enumerate(text_pos_list):
        for alpha in alpha_array:
            figure.clear()
            text_axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
            text_axes.axis("off")

            if idx > 0:
                text_axes.text(
                    LEFT_ALIGN,
                    0,
                    s=sentence[: text_pos_list[idx - 1]],
                    fontsize=48,
                    style="oblique",
                    ha="left",
                    va="bottom",
                    color="white",
                    alpha=1.0,
                )

            text_axes.text(
                LEFT_ALIGN,
                0,
                s=sentence[:text_pos],
                fontsize=48,
                style="oblique",
                ha="left",
                va="bottom",
                color="white",
                alpha=alpha,
            )
            im = convert_plot_to_image(figure)
            yield im

    # Keep the image for this many frames
    for i in range(persist_frames):
        yield im

    # Fade out the image over this many frames
    fade_out_alpha = np.power(np.linspace(1, 0, fade_out_frames), 2)
    for alpha in fade_out_alpha:
        figure.clear()
        text_axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        text_axes.axis("off")
        text_axes.text(
            LEFT_ALIGN,
            0,
            s=sentence,
            fontsize=48,
            style="oblique",
            ha="left",
            va="bottom",
            color="white",
            alpha=alpha,
        )
        im = convert_plot_to_image(figure)
        yield im

    # Stay black for the remainder
    black_screen = np.array(im)
    black_screen[:, :, :] = 0
    im = Image.fromarray(black_screen)
    while True:
        yield im


@dataclass
class FireAutomata:
    height: int
    width: int
    decay: float
    spawn_points: int
    heatmap: np.ndarray = field(init=False)
    spawn_indices: np.ndarray = field(init=False)
    non_spawn_indices: np.ndarray = field(init=False)
    flame_base: np.ndarray = field(init=False)
    height_max_index: int = field(init=False)
    width_max_index: int = field(init=False)

    def __post_init__(self):
        self.heatmap = np.zeros((self.height, self.width))
        indices = np.arange(self.width)
        self.spawn_indices = np.random.choice(indices, 20)
        self.non_spawn_indices = np.delete(indices, self.spawn_points)
        self.flame_base = np.zeros(self.width)
        self.height_max_index = self.height - 1
        self.width_max_index = self.width - 1

    def update_heatmap(self):
        swap_spawn = np.random.randint(len(self.spawn_indices))
        swap_non_spawn = np.random.randint(len(self.non_spawn_indices))
        self.spawn_indices[swap_spawn], self.non_spawn_indices[swap_non_spawn] = (
            self.non_spawn_indices[swap_non_spawn],
            self.spawn_indices[swap_spawn],
        )
        self.flame_base *= 0
        self.flame_base[self.spawn_indices] = 1
        self.heatmap[self.height_max_index, :] = self.flame_base

        delta = np.random.random((self.height_max_index, self.width, 3))
        delta[:, self.width_max_index, 0] = 0
        delta[:, 0, 2] = 0
        scaled_delta = delta / delta.sum(axis=2)[:, :, np.newaxis]
        heatmap_source_part = np.zeros((self.height_max_index, self.width, 3))
        heatmap_source_part[:, : self.width_max_index, 0] = self.heatmap[
            1 : self.height, 1 : self.width
        ]
        heatmap_source_part[:, :, 1] = self.heatmap[1 : self.height, :]
        heatmap_source_part[:, 1 : self.width, 2] = self.heatmap[
            1 : self.height, : self.width_max_index
        ]
        self.heatmap[: self.height_max_index, :] = (
            heatmap_source_part * scaled_delta
        ).sum(axis=2) * self.decay


def draw_fire_automata(
    axes_dims: List[float],
    fade_in_frames: int,
    update_frames: int,
    fade_out_frames: int,
) -> Generator[Image.Image, None, None]:
    im: Image.Image = Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8))
    fire_automata = FireAutomata(height=65, width=64, decay=0.95, spawn_points=20)
    figure = plt.figure(figsize=(19.2, 10.8))

    fade_in_alpha = np.power(np.linspace(0, 1, fade_in_frames), 2)
    for alpha in fade_in_alpha:
        figure.clear()
        render_axes = figure.add_axes(axes_dims)
        fire_automata.update_heatmap()
        render_axes.imshow(
            fire_automata.heatmap[:-1, :],
            cmap="hot",
            interpolation="nearest",
            alpha=alpha,
        )
        render_axes.axis("off")
        im = convert_plot_to_image(figure)
        yield im

    for frame_number in range(update_frames):
        figure.clear()
        render_axes = figure.add_axes(axes_dims)
        fire_automata.update_heatmap()
        render_axes.imshow(
            fire_automata.heatmap[:-1, :], cmap="hot", interpolation="nearest"
        )
        render_axes.axis("off")
        im = convert_plot_to_image(figure)
        yield im

    fade_out_alpha = np.power(np.linspace(1, 0, fade_out_frames), 2)
    for alpha in fade_out_alpha:
        figure.clear()
        render_axes = figure.add_axes(axes_dims)
        fire_automata.update_heatmap()
        render_axes.imshow(
            fire_automata.heatmap[:-1, :],
            cmap="hot",
            interpolation="nearest",
            alpha=alpha,
        )
        render_axes.axis("off")
        im = convert_plot_to_image(figure)
        yield im

    # Stay black for the remainder
    black_screen = np.array(im)
    black_screen[:, :, :] = 0
    im = Image.fromarray(black_screen)
    while True:
        yield im


def draw_gaussian(
    axes_dims: List[float],
    fade_in_frames: int,
    update_frames: int,
    persist_frames: int,
    fade_out_frames: int,
) -> Generator[Image.Image, None, None]:
    figure = plt.figure(figsize=(19.2, 10.8))
    with plt.style.context("dark_background"):
        ax = figure.add_axes(axes_dims)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.set_xlim((-8.8, 8.8))
        ax.set_ylim((-0.02, 0.42))
    im = convert_plot_to_image(figure)
    # [0.05, 0.1, 0.9, 0.25]

    # Fade in the axes over this many frames
    fade_out_alpha = np.power(np.linspace(0, 1, fade_in_frames), 2)
    for alpha in fade_out_alpha:
        pixels = np.array(im)
        alpha_layer = pixels[:, :, 3]
        alpha_layer[alpha_layer > 0] = int(255 * alpha)
        yield Image.fromarray(pixels)

    # Animate the Guassian
    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    for frame in range(0, update_frames * 4, 4):
        figure.clear()
        with plt.style.context("dark_background"):
            ax = figure.add_axes(axes_dims)
            x = np.linspace(mu - 8 * sigma, mu + 8 * sigma, update_frames * 4)
            ax.plot(
                x[:frame],
                stats.norm.pdf(x[:frame], mu, sigma),
                linewidth=3,
                color="skyblue",
            )
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)
            ax.set_xlim((-8.8, 8.8))
            ax.set_ylim((-0.02, 0.42))
        im = convert_plot_to_image(figure)
        yield im

    for frame in range(persist_frames):
        yield im

    # Fade out the image over this many frames
    fade_out_alpha = np.power(np.linspace(1, 0, fade_out_frames), 2)
    for alpha in fade_out_alpha:
        pixels = np.array(im)
        alpha_layer = pixels[:, :, 3]
        alpha_layer[alpha_layer > 0] = int(255 * alpha)
        yield Image.fromarray(pixels)

    # Stay black for the remainder
    black_screen = np.array(im)
    black_screen[:, :, :] = 0
    im = Image.fromarray(black_screen)
    while True:
        yield im


def main():
    anim_file_path = Path("./test.mp4")
    figure = plt.figure(figsize=(19.2, 10.8))

    file_writer = FFMpegFileWriter(fps=FRAME_RATE)
    with file_writer.saving(figure, anim_file_path, dpi=100):
        intro_text = Scene(
            0,
            169,
            1,
            draw_text(
                sentence="I have seen things you people would not believe",
                text_pos_list=[19, 47],
                alpha_transitions=60,
                persist_frames=0,
                fade_out_frames=24,
            ),
        )
        eye = Scene(
            0,
            193,
            0,
            draw_eye(
                axes_dims=[0, 0.22, 1.0, 0.8], persist_frames=24, fade_out_frames=24
            ),
        )
        heatmap = Scene(
            121,
            313,
            2,
            draw_fire_automata(
                axes_dims=[0.2, 0.35, 0.6, 0.6],
                fade_in_frames=24,
                update_frames=144,
                fade_out_frames=24,
            ),
        )
        gaussian = Scene(
            169,
            313,
            1,
            draw_gaussian(
                axes_dims=[0.05, 0.1, 0.9, 0.25],
                fade_in_frames=24,
                update_frames=72,
                persist_frames=24,
                fade_out_frames=24,
            ),
        )
        heatmaps_text = Scene(
            145,
            313,
            1,
            draw_text(
                sentence="Heat maps on fire off the shoulder of a Gaussian",
                text_pos_list=[17, 48],
                alpha_transitions=60,
                persist_frames=24,
                fade_out_frames=24,
            ),
        )
        active_scenes_list: List[Scene] = [
            intro_text,
            eye,
            heatmap,
            gaussian,
            heatmaps_text,
        ]
        active_scenes_list.sort(key=lambda scene: scene.zorder, reverse=True)

        for frame_number in itertools.count():
            figure.clear()
            render_axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
            render_axes.axis("off")
            active_scene_count = len(active_scenes_list)
            if active_scene_count <= 0:
                break
            rendered_scene: bool = False
            for scene_index in range(active_scene_count - 1, -1, -1):
                scene = active_scenes_list[scene_index]
                if frame_number >= scene.start_frame:
                    if frame_number > scene.end_frame:
                        del active_scenes_list[scene_index]
                    else:
                        render_axes.imshow(next(scene.render_frame))
                        rendered_scene = True
            if rendered_scene is True:
                file_writer.grab_frame(facecolor=BACKGROUND_COLOUR)


if __name__ == "__main__":
    main()
