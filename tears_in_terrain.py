import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegFileWriter
from pathlib import Path

BACKGROUND_COLOUR = "#000000FF"
FRAME_RATE = 24

# class


def main():
    anim_file_path = Path(".")
    figure = plt.figure(figsize=(19.2, 10.8))
    # map_axes = figure.add_axes([0.0, 0.0, 1.0, 0.6])

    file_writer = FFMpegFileWriter(fps=FRAME_RATE)
    with file_writer.saving(figure, anim_file_path, dpi=100):

        file_writer.grab_frame(facecolor=BACKGROUND_COLOUR)
        pass


if __name__ == "__main__":
    main()
