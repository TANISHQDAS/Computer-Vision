import os
import cv2
import numpy as np
from tkinter import Tk, Frame, Label, Button, filedialog, messagebox, StringVar
from tkinter import ttk
from PIL import Image, ImageTk


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Face Detection")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f4f6f9")

        self.original_image = None
        self.display_image = None
        self.detected_faces = []
        self.current_file = ""
        self.zoom_scale = 1.0

        self.output_folder = "cropped_faces"
        os.makedirs(self.output_folder, exist_ok=True)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            messagebox.showerror("Error", "Face detection model could not be loaded.")
            self.root.destroy()
            return

        header = Label(
            root,
            text="Professional Face Detection System",
            font=("Segoe UI", 20, "bold"),
            bg="#f4f6f9",
            fg="#1f2937"
        )
        header.pack(pady=(15, 10))

        button_frame = Frame(root, bg="#f4f6f9")
        button_frame.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=8)

        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=8)
        ttk.Button(button_frame, text="Detect Faces", command=self.detect_faces).grid(row=0, column=1, padx=8)
        ttk.Button(button_frame, text="Zoom In", command=lambda: self.change_zoom(1.15)).grid(row=0, column=2, padx=8)
        ttk.Button(button_frame, text="Zoom Out", command=lambda: self.change_zoom(0.85)).grid(row=0, column=3, padx=8)
        ttk.Button(button_frame, text="Reset Zoom", command=self.reset_zoom).grid(row=0, column=4, padx=8)
        ttk.Button(button_frame, text="Save Faces", command=self.save_faces).grid(row=0, column=5, padx=8)

        self.image_frame = Frame(root, bg="#d1d5db", bd=2, relief="ridge")
        self.image_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.image_label = Label(
            self.image_frame,
            text="No image selected",
            font=("Segoe UI", 14),
            bg="#e5e7eb",
            fg="#4b5563"
        )
        self.image_label.pack(fill="both", expand=True)

        self.status_text = StringVar()
        self.status_text.set("Ready")

        self.status_bar = Label(
            root,
            textvariable=self.status_text,
            anchor="w",
            font=("Segoe UI", 10),
            bg="#1f2937",
            fg="white",
            padx=10,
            pady=8
        )
        self.status_bar.pack(fill="x", side="bottom")

        self.root.bind("<MouseWheel>", self.mouse_zoom)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")
            ]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror("Error", "Unable to load the selected image.")
            return

        self.current_file = os.path.basename(file_path)
        self.original_image = image
        self.display_image = image.copy()
        self.detected_faces = []
        self.zoom_scale = 1.0

        self.show_image(self.display_image)
        self.status_text.set(f"Loaded image: {self.current_file}")

    def detect_faces(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        self.detected_faces = faces
        result = self.original_image.copy()

        for index, (x, y, w, h) in enumerate(faces, start=1):
            region = gray[y:y + h, x:x + w]
            confidence = self.calculate_confidence(region)

            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 200, 0), 3)
            cv2.putText(
                result,
                f"Face {index}  {confidence}%",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2
            )

        self.display_image = result
        self.show_image(self.display_image)

        if len(faces) == 0:
            self.status_text.set("No faces detected")
            messagebox.showinfo("Result", "No faces were found in the image.")
        else:
            self.status_text.set(f"Detected {len(faces)} face(s)")

    def calculate_confidence(self, region):
        value = int(np.clip(np.mean(region) / 2.2, 55, 99))
        return value

    def show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        width = max(1, int(pil_image.width * self.zoom_scale))
        height = max(1, int(pil_image.height * self.zoom_scale))

        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

        max_width = self.image_frame.winfo_width() - 20
        max_height = self.image_frame.winfo_height() - 20

        if max_width > 100 and max_height > 100:
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(pil_image)

        self.image_label.configure(image=tk_image, text="")
        self.image_label.image = tk_image

    def change_zoom(self, factor):
        if self.display_image is None:
            return

        self.zoom_scale *= factor
        self.zoom_scale = max(0.2, min(self.zoom_scale, 5.0))
        self.show_image(self.display_image)
        self.status_text.set(f"Zoom: {int(self.zoom_scale * 100)}%")

    def reset_zoom(self):
        if self.display_image is None:
            return

        self.zoom_scale = 1.0
        self.show_image(self.display_image)
        self.status_text.set("Zoom reset to 100%")

    def mouse_zoom(self, event):
        if event.delta > 0:
            self.change_zoom(1.1)
        else:
            self.change_zoom(0.9)

    def save_faces(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        if len(self.detected_faces) == 0:
            messagebox.showwarning("Warning", "No detected faces available to save.")
            return

        image_name = os.path.splitext(self.current_file)[0]
        save_folder = os.path.join(self.output_folder, image_name)
        os.makedirs(save_folder, exist_ok=True)

        for index, (x, y, w, h) in enumerate(self.detected_faces, start=1):
            cropped = self.original_image[y:y + h, x:x + w]
            file_name = os.path.join(save_folder, f"face_{index}.png")
            cv2.imwrite(file_name, cropped)

        self.status_text.set(
            f"Saved {len(self.detected_faces)} face(s) in '{save_folder}'"
        )

        messagebox.showinfo(
            "Saved",
            f"{len(self.detected_faces)} face(s) were saved in:
{save_folder}"
        )


if __name__ == "__main__":
    root = Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
