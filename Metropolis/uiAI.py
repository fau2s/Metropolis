import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Trading UI")
root.geometry("1200x700")  # set your preferred window size

# --- Notebook ---
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# --- Frames (Tabs) ---
frames = [ttk.Frame(notebook) for _ in range(8)]
tab_names = ["Trading", "Stocks", "VAR", "Seq2Seq train", "Hilbert curve",
              "Constituents", "Filter portfolio", "Return distributions"]

for frame, name in zip(frames, tab_names):
    notebook.add(frame, text=name)

# --- Example for 'Trading' Tab (frame1) ---
frame1 = frames[0]
lf_trading = ttk.LabelFrame(frame1, text="Trading Controls")
lf_trading.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# Labels and entries in a grid
labels = ["Input 1:", "Input 2:", "Input 3:"]
entries = []

for i, text in enumerate(labels):
    label = ttk.Label(lf_trading, text=text)
    label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    entry = ttk.Entry(lf_trading, width=15)
    entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
    entries.append(entry)

# Buttons in the same LabelFrame
btn_run = ttk.Button(lf_trading, text="Run Trading", command=lambda: print("Run Trading"))
btn_run.grid(row=0, column=2, padx=10, pady=5)

btn_save = ttk.Button(lf_trading, text="Save Image", command=lambda: print("Save Image"))
btn_save.grid(row=1, column=2, padx=10, pady=5)

# Progress bar below
progress = ttk.Progressbar(lf_trading, orient="horizontal", length=200, mode="determinate")
progress.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

# --- Example for 'VAR' Tab (frame3) ---
frame3 = frames[2]
lf_var = ttk.LabelFrame(frame3, text="VAR fitting")
lf_var.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

label_var = ttk.Label(lf_var, text="Input 4:")
label_var.grid(row=0, column=0, padx=5, pady=2)

input_var = ttk.Entry(lf_var, width=15)
input_var.grid(row=0, column=1, padx=5, pady=2)

btn_fit = ttk.Button(lf_var, text="Fit VAR(p)", command=lambda: print("Fit VAR"))
btn_fit.grid(row=1, column=0, columnspan=2, pady=5)

# --- Example for 'Stocks' Tab (frame2) ---
frame2 = frames[1]
btn_plot_stocks = ttk.Button(frame2, text="Plot Stocks", command=lambda: print("Plot Stocks"))
btn_plot_stocks.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# --- Example for 'Return distributions' Tab (frame8) ---
frame8 = frames[7]
lf_return = ttk.LabelFrame(frame8, text="Return Distributions")
lf_return.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# Example: Label + Entry for 'ratio'
label_ratio = ttk.Label(lf_return, text="Ratio:")
label_ratio.grid(row=0, column=0, padx=5, pady=2)

entry_ratio = ttk.Entry(lf_return, width=15)
entry_ratio.grid(row=0, column=1, padx=5, pady=2)

# --- Example: Dropdown in frame2 ---
selected = tk.StringVar()
options = [str(i) for i in range(400)]
drop = ttk.OptionMenu(frame2, selected, options[0], *options)
drop.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

# --- Example: Checkbuttons in a LabelFrame ---
lf_checkbuttons = ttk.LabelFrame(lf_trading, text="Options")
lf_checkbuttons.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")

options_texts = ["VAR1", "Trading Range", "Online LSTM", "Offline LSTM", "Seq2Seq LSTM", "NLP thread LSTM"]
check_vars = []

for i, text in enumerate(options_texts):
    var = tk.StringVar(value="")
    cb = ttk.Checkbutton(lf_checkbuttons, text=text, variable=var, onvalue=text, offvalue="")
    cb.grid(row=i//2, column=i%2, sticky="w", padx=5, pady=2)
    check_vars.append(var)

root.mainloop()
