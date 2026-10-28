import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# Dummy data generator (NxL matrix)
def generate_dummy_data(N, L):
    np.random.seed(0)
    return np.cumsum(np.random.randn(N, L), axis=1)

class VARApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VAR Model Calibration Video")

        # Input frame
        input_frame = ttk.LabelFrame(root, text="Inputs")
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ttk.Label(input_frame, text="Calibration Length (l):").grid(row=0, column=0, padx=5, pady=5)
        self.calib_length_entry = ttk.Entry(input_frame, width=10)
        self.calib_length_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Calibration Start (s):").grid(row=1, column=0, padx=5, pady=5)
        self.calib_start_entry = ttk.Entry(input_frame, width=10)
        self.calib_start_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(input_frame, text="Run Calibration", command=self.run_calibration).grid(row=2, column=0, columnspan=2, pady=10)

        # Plot frame
        plot_frame = ttk.LabelFrame(root, text="Regression Matrix Video")
        plot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Dummy data
        self.N, self.L = 3, 200
        self.data = generate_dummy_data(self.N, self.L)

    def run_calibration(self):
        try:
            l = int(self.calib_length_entry.get())
            s = int(self.calib_start_entry.get())
        except ValueError:
            print("Please enter valid integers for calibration length and start.")
            return

        if l <= 0 or s < 0 or s + l >= self.L:
            print("Invalid l or s values. Ensure 0 <= s and s + l < L.")
            return

        # Calibration video frames
        self.frames = []
        for t in range(s, self.L - l):
            window_data = self.data[:, t:t + l].T  # shape (l, N)
            df_window = pd.DataFrame(window_data)
            model = VAR(df_window)
            results = model.fit(maxlags=1)
            reg_matrix = results.coefs[0]  # shape (N, N)

            # Plotting regression matrix
            fig, ax = plt.subplots(figsize=(4, 4))
            cax = ax.matshow(reg_matrix, cmap='viridis')
            ax.set_title(f"Time {t}")
            plt.colorbar(cax, ax=ax)
            plt.close(fig)  # Close to avoid memory leaks

            self.frames.append(fig)

        # Start animation
        self.ani = animation.FuncAnimation(self.figure, self.update_frame, frames=len(self.frames), interval=500, repeat=True)
        self.canvas.draw()

    def update_frame(self, i):
        self.figure.clf()  # Clear the previous plot
        # Draw current frame (regression matrix figure)
        frame_fig = self.frames[i]
        # Transfer the frame's Axes image to the main figure
        ax_main = self.figure.add_subplot(111)
        ax_frame = frame_fig.axes[0]

        # Transfer matrix image
        im = ax_frame.images[0]
        ax_main.imshow(im.get_array(), cmap=im.get_cmap())
        ax_main.set_title(ax_frame.get_title())

        # Transfer colorbar
        self.figure.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)

if __name__ == "__main__":
    root = tk.Tk()
    app = VARApp(root)
    root.mainloop()
