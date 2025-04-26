import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

model_names = ["AB", "RF", "LDA", "KNN", "SVM", "DT", "GNB", "LR"]

models = {}


# Load trained models
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


for name in model_names:
    with open(resource_path(f"models\\{name}_model.pkl"), "rb") as f:
        models[name] = pickle.load(f)


# Load preprocessor and processed column names
with open(resource_path("preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)


with open(resource_path("template_columns.pkl"), "rb") as f:
    template_columns = pickle.load(f)


# Define expected raw input columns
raw_input_columns = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "jaundice",
]

example_choices = {
    "gender": ["m", "f"],
    "jaundice": ["yes", "no"],
    "age": ["1 to 100 in years"],
}

# GUI setup
root = tk.Tk()
root.title("Autism Prediction")
root.geometry("550x650")
entries = {}


def on_closing():
    root.destroy()
    sys.exit()  # or use os._exit(0) if sys.exit() still hangs


root.protocol("WM_DELETE_WINDOW", on_closing)


def predict():
    try:
        user_input = {}
        for col in raw_input_columns:
            val = entries[col].get().lower().strip()
            if col in example_choices and col != "age":
                if val not in example_choices[col]:
                    messagebox.showerror(
                        "Input Error", f"Invalid value for {col}: {val}"
                    )
                    return
                user_input[col] = val
            else:
                try:
                    user_input[col] = int(val) if col.startswith("A") else float(val)
                except ValueError:
                    messagebox.showerror(
                        "Input Error", f"Invalid number for {col}: {val}"
                    )
                    return

        df = pd.DataFrame([user_input])
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.lower().str.strip()
        transformed = preprocessor.transform(df)

        results = []
        label_counts = {"Autistic": 0, "Not Autistic": 0}
        confidence_scores = {}

        # Scrollable Result Window Setup
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Results")
        result_window.geometry("750x800")
        result_window.configure(bg="#f7f7f7")

        canvas = tk.Canvas(result_window, bg="#f7f7f7", width=780, height=1150)
        scrollbar = ttk.Scrollbar(
            result_window, orient="vertical", command=canvas.yview
        )
        scrollable_frame = tk.Frame(canvas, bg="#f7f7f7")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable scrolling with mousewheel anywhere inside the result window
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Add model prediction results
        frame = tk.Frame(scrollable_frame, bg="#f7f7f7")
        frame.pack(pady=10)

        for name, model in models.items():
            pred = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0]
            confidence = max(prob) * 100
            label = "Autistic" if pred == "yes" else "Not Autistic"
            label_counts[label] += 1
            confidence_scores[name] = confidence
            color = "red" if label == "Autistic" else "green"
            label_text = f"{name}: {label} ({confidence:.1f}% confident)"
            tk.Label(
                frame,
                text=label_text,
                font=("Arial", 11, "bold"),
                fg=color,
                bg="#f7f7f7",
            ).pack(pady=2)

        # Pie Chart: Distribution
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(
            label_counts.values(),
            labels=label_counts.keys(),
            autopct="%1.1f%%",
            colors=["red", "green"],
        )
        ax1.set_title("Prediction Distribution")
        pie_chart = FigureCanvasTkAgg(fig1, master=scrollable_frame)
        pie_chart.get_tk_widget().pack(pady=15)
        pie_chart.draw()

        # Bar Chart: Confidence Scores
        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
        ax2.bar(confidence_scores.keys(), confidence_scores.values(), color="skyblue")
        ax2.set_title("Model Confidence Scores")
        ax2.set_ylabel("Confidence (%)")
        ax2.set_ylim(0, 110)
        for i, (model, score) in enumerate(confidence_scores.items()):
            ax2.text(i, score + 1, f"{score:.1f}%", ha="center", fontsize=9)
        bar_chart = FigureCanvasTkAgg(fig2, master=scrollable_frame)
        bar_chart.get_tk_widget().pack(pady=15)
        bar_chart.draw()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Input fields
row = 0
for col in raw_input_columns:
    label_text = (
        f"{col} ({', '.join(example_choices[col])})"
        if col in example_choices
        else f"{col} (0 - yes/ 1 - no)"
    )
    ttk.Label(root, text=label_text, font=("Arial", 10)).grid(
        row=row, column=0, sticky="w", pady=5, padx=5
    )
    entry = ttk.Entry(root, width=40)
    entry.grid(row=row, column=1, padx=10, pady=5)
    entries[col] = entry
    row += 1

# Predict Button
ttk.Button(root, text="Predict Autism Status", command=predict).grid(
    row=row, column=0, columnspan=2, pady=20
)

root.mainloop()
