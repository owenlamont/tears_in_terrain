from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegFileWriter

BACKGROUND_COLOUR = "#000000FF"
FRAME_RATE = 24
LEFT_ALIGN = 0.08


def main():
    anim_file_path = Path("./test.mp4")
    figure = plt.figure(figsize=(19.2, 10.8))
    alpha_array = np.power(np.linspace(0, 1, 50), 2)

    file_writer = FFMpegFileWriter(fps=FRAME_RATE)
    with file_writer.saving(figure, anim_file_path, dpi=100):
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
                        s=sentence[:text_pos_list[idx-1]],
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
                file_writer.grab_frame(facecolor=BACKGROUND_COLOUR)


if __name__ == "__main__":
    main()
