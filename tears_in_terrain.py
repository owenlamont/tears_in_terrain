import io
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.animation import FFMpegFileWriter
import matplotlib.patches as patches

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


def draw_eye(axes_dims: List[float]) -> Generator[Image.Image, None, None]:
    interval_count = 361
    angle = np.linspace(0, np.pi * 2.0, interval_count)
    radius = np.array([num % 2 for num in range(0, interval_count)]) * 2.5 + 1.5
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    iris = np.vstack([x.reshape(1, -1), y.reshape(1, -1)])
    intervals = np.linspace(-7.05, 7.05, interval_count)
    positive_curve = 0.075 * intervals ** 2 - 3.75
    negative_curve = -0.075 * (intervals ** 2) + 3.75
    im: Optional[Image.Image] = None
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

    while True:
        yield im


def draw_text(
    sentence: str, text_pos_list: List[int], alpha_transitions: int
) -> Generator[Image.Image, None, None]:
    """
    Render the next frame
    :return: The next render as a PIL Image
    """
    im: Optional[Image.Image] = None
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

    while True:
        yield im


def draw_fire_automata(axes_dims: List[float]) -> Generator[Image.Image, None, None]:
    WIDTH = 64
    WIDTH_MAX_INDEX = WIDTH - 1
    HEIGHT = 65
    HEIGHT_MAX_INDEX = HEIGHT - 1
    DECAY = 0.95

    indices = np.arange(WIDTH)
    spawn_indices = np.random.choice(indices, 20)
    non_spawn_indices = np.delete(indices, spawn_indices)

    figure = plt.figure(figsize=(19.2, 10.8))
    heatmap = np.zeros((HEIGHT, WIDTH))
    flame_base = np.zeros(WIDTH)

    for frame_number in range(200):
        figure.clear()
        render_axes = figure.add_axes(axes_dims)

        swap_spawn = np.random.randint(len(spawn_indices))
        swap_non_spawn = np.random.randint(len(non_spawn_indices))
        spawn_indices[swap_spawn], non_spawn_indices[swap_non_spawn] = (
            non_spawn_indices[swap_non_spawn],
            spawn_indices[swap_spawn],
        )
        flame_base *= 0
        flame_base[spawn_indices] = 1
        heatmap[HEIGHT_MAX_INDEX, :] = flame_base

        delta = np.random.random((HEIGHT_MAX_INDEX, WIDTH, 3))
        delta[:, WIDTH_MAX_INDEX, 0] = 0
        delta[:, 0, 2] = 0
        scaled_delta = delta / delta.sum(axis=2)[:, :, np.newaxis]
        heatmap_source_part = np.zeros((HEIGHT_MAX_INDEX, WIDTH, 3))
        heatmap_source_part[:, :WIDTH_MAX_INDEX, 0] = heatmap[1:HEIGHT, 1:WIDTH]
        heatmap_source_part[:, :, 1] = heatmap[1:HEIGHT, :]
        heatmap_source_part[:, 1:WIDTH, 2] = heatmap[1:HEIGHT, :WIDTH_MAX_INDEX]
        heatmap[:HEIGHT_MAX_INDEX, :] = (heatmap_source_part * scaled_delta).sum(
            axis=2
        ) * DECAY

        render_axes.imshow(
            heatmap[:HEIGHT_MAX_INDEX, :], cmap="hot", interpolation="nearest"
        )
        render_axes.axis("off")

        im = convert_plot_to_image(figure)
        yield im


def main():
    anim_file_path = Path("./test.mp4")
    figure = plt.figure(figsize=(19.2, 10.8))

    file_writer = FFMpegFileWriter(fps=FRAME_RATE)
    with file_writer.saving(figure, anim_file_path, dpi=100):
        intro_text = Scene(
            0,
            121,
            1,
            draw_text("I have seen things you people would not believe", [19, 47], 60),
        )
        eye = Scene(0, 121, 0, draw_eye(axes_dims=[0, 0.22, 1.0, 0.8]))
        heatmap = Scene(0, 121, 2, draw_fire_automata(axes_dims=[0.2, 0.35, 0.6, 0.6]))
        active_scenes_list: List[Scene] = [intro_text, eye, heatmap]
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
