import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from tensorflow.keras.datasets import mnist

# ─────────────────────────────────────────────
# Exercise 1 – Load MNIST and pick digit "6"
# ─────────────────────────────────────────────
(x_train, y_train), _ = mnist.load_data()

# Grab the first occurrence of the digit 6 (28x28 grayscale array, values 0-255)
index_of_6 = np.where(y_train == 6)[0][0]
digit_6 = x_train[index_of_6]          # shape: (28, 28)

# ─────────────────────────────────────────────
# Load the base image and build a 128x128 canvas
# ─────────────────────────────────────────────
img = Image.open("p1.jpg").resize((128, 128)).convert("L")
canvas_base = np.array(img, dtype=np.float32)   # shape: (128, 128)


# ─────────────────────────────────────────────
# Exercise 2 – Insert one digit at pixel (i, j)
# ─────────────────────────────────────────────
def insert_digit(canvas: np.ndarray, digit: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Overlay `digit` (28x28) onto `canvas` (128x128) with its top-left corner
    at pixel position (row=i, col=j).

    Parameters
    ----------
    canvas : np.ndarray  – 128x128 grayscale background image
    digit  : np.ndarray  – 28x28 MNIST digit patch
    i      : int         – row index  (0 … 128)
    j      : int         – column index (0 … 128)

    Returns
    -------
    np.ndarray – copy of canvas with the digit pasted in
    """
    h, w = digit.shape          # always 28, 28

    # Clamp the paste region so we never go out of bounds
    row_end = min(i + h, canvas.shape[0])
    col_end = min(j + w, canvas.shape[1])
    crop_h  = row_end - i
    crop_w  = col_end - j

    if crop_h <= 0 or crop_w <= 0:
        print(f"[insert_digit] position ({i},{j}) is completely outside the canvas.")
        return canvas

    result = canvas.copy()
    result[i:row_end, j:col_end] = digit[:crop_h, :crop_w]
    return result


# ─────────────────────────────────────────────
# Exercise 3 – Linear trajectory simulation
# ─────────────────────────────────────────────
def simulate_linear(canvas: np.ndarray,
                    digit: np.ndarray,
                    start: tuple[int, int],
                    end: tuple[int, int],
                    steps: int = 30,
                    save_gif: str = "linear_trajectory.gif") -> list[np.ndarray]:
    """
    Move `digit` from `start` to `end` along a straight (linear) path.

    The position at step k is:
        row(k) = start_row + k * (end_row - start_row) / (steps - 1)
        col(k) = start_col + k * (end_col - start_col) / (steps - 1)

    Parameters
    ----------
    canvas   : 128x128 background
    digit    : 28x28 MNIST digit
    start    : (row, col) starting position
    end      : (row, col) ending position
    steps    : number of animation frames
    save_gif : filename to save the GIF animation

    Returns
    -------
    list of np.ndarray frames
    """
    frames = []
    for k in range(steps):
        t = k / (steps - 1)
        row = int(round(start[0] + t * (end[0] - start[0])))
        col = int(round(start[1] + t * (end[1] - start[1])))
        frame = insert_digit(canvas, digit, row, col)
        frames.append(frame)

    _save_animation(frames, save_gif, title="Linear Trajectory")
    return frames


# ─────────────────────────────────────────────
# Exercise 4 – Quadratic trajectory simulation
# ─────────────────────────────────────────────
def simulate_quadratic(canvas: np.ndarray,
                       digit: np.ndarray,
                       start: tuple[int, int],
                       control: tuple[int, int],
                       end: tuple[int, int],
                       steps: int = 30,
                       save_gif: str = "quadratic_trajectory.gif") -> list[np.ndarray]:
    """
    Move `digit` along a quadratic Bézier curve defined by three points:
        start   – P0, where the digit begins
        control – P1, the "pull" point that creates the curve
        end     – P2, where the digit ends

    Bézier formula (t in [0,1]):
        B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2

    Parameters
    ----------
    canvas   : 128x128 background
    digit    : 28x28 MNIST digit
    start    : (row, col) P0
    control  : (row, col) P1 – control / apex point
    end      : (row, col) P2
    steps    : number of animation frames
    save_gif : filename to save the GIF animation

    Returns
    -------
    list of np.ndarray frames
    """
    frames = []
    for k in range(steps):
        t  = k / (steps - 1)
        u  = 1 - t
        row = int(round(u**2 * start[0]   + 2*u*t * control[0]   + t**2 * end[0]))
        col = int(round(u**2 * start[1]   + 2*u*t * control[1]   + t**2 * end[1]))
        frame = insert_digit(canvas, digit, row, col)
        frames.append(frame)

    _save_animation(frames, save_gif, title="Quadratic (Bézier) Trajectory")
    return frames


# ─────────────────────────────────────────────
# Helper – save a list of frames as an animated GIF
# ─────────────────────────────────────────────
def _save_animation(frames: list[np.ndarray], filename: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    if title:
        ax.set_title(title)

    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255, animated=True)

    def update(frame_data):
        im.set_data(frame_data)
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=80, blit=True
    )
    ani.save(filename, writer="pillow")
    plt.close(fig)
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────
# Demo – run everything and show results
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # --- Exercise 2: single insert ---
    result_single = insert_digit(canvas_base, digit_6, i=50, j=50)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(canvas_base, cmap="gray"); axes[0].set_title("Original canvas"); axes[0].axis("off")
    axes[1].imshow(digit_6,     cmap="gray"); axes[1].set_title("Digit 6 (MNIST)"); axes[1].axis("off")
    axes[2].imshow(result_single, cmap="gray"); axes[2].set_title("Digit 6 @ (50,50)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("exercise2_insert.png", dpi=120)
    plt.show()
    print("Saved: exercise2_insert.png")

    # --- Exercise 3: linear trajectory ---
    # digit travels from top-left to bottom-right
    linear_frames = simulate_linear(
        canvas_base, digit_6,
        start=(5, 5),
        end=(90, 90),
        steps=40,
        save_gif="linear_trajectory.gif"
    )

    # --- Exercise 4: quadratic (Bezier) trajectory ---
    # digit arcs from left side, peaks at the top-centre, lands on the right
    quadratic_frames = simulate_quadratic(
        canvas_base, digit_6,
        start=(90, 5),
        control=(5, 50),
        end=(90, 90),
        steps=40,
        save_gif="quadratic_trajectory.gif"
    )

    print("\nAll exercises complete!")
    print("  exercise2_insert.png     - static insert demo")
    print("  linear_trajectory.gif    - Exercise 3 animation")
    print("  quadratic_trajectory.gif - Exercise 4 animation")
