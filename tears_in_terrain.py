from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegFileWriter
import io
from PIL import Image
from dataclasses import dataclass
from typing import Optional
import itertools

BACKGROUND_COLOUR = "#000000FF"
FRAME_RATE = 24
LEFT_ALIGN = 0.08


@dataclass
class IntroText:
    """
    Class for rendering the intro text
    """

    start_frame: int
    render_order: int

    def __iter__(self) -> Image:
        """
        Render the next frame
        :return: The next render as a PIL Image
        """
        figure = plt.figure(figsize=(19.2, 10.8))
        alpha_array = np.power(np.linspace(0, 1, 50), 2)
        text_pos_list = [19, 47]
        sentence = "I have seen things you people would not believe"
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
                buf = io.BytesIO()
                figure.savefig(buf, format="png", facecolor="black")
                buf.seek(0)
                im = Image.open(buf)
                yield im


def main():
    anim_file_path = Path("./test.mp4")
    figure = plt.figure(figsize=(19.2, 10.8))

    file_writer = FFMpegFileWriter(fps=FRAME_RATE)
    with file_writer.saving(figure, anim_file_path, dpi=100):
        intro_text = IntroText(0,0)
        active_scenes_list = [(iter(intro_text), intro_text.start_frame)]
        for frame_number in itertools.count():
            figure.clear()
            render_axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
            render_axes.axis("off")
            active_scene_count = len(active_scenes_list)
            if active_scene_count <= 0:
                break
            rendered_scene: bool = False
            for scene_index in range(active_scene_count-1,-1,-1):
                scene = active_scenes_list[scene_index]
                if frame_number >= scene[1]:
                    try:
                        render_axes.imshow(next(scene[0]))
                        rendered_scene = True
                    except StopIteration:
                        del active_scenes_list[scene_index]
            if rendered_scene is True:
                file_writer.grab_frame(facecolor=BACKGROUND_COLOUR)


if __name__ == "__main__":
    main()
