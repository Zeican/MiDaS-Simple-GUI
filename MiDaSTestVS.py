import sys
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import PySimpleGUI as sg
from PIL import ImageTk

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the depth estimation models
models = {
    "MiDaS Small": "MiDaS_small",
    "MiDaS Large": "DPT_Large",
    "MiDaS Hybrid": "DPT_Hybrid",
}

# Default downscaled size and model
default_downscaled_size = (512, 512)
default_model = "MiDaS Hybrid"

# Load the selected depth estimation model
def load_depth_model(model_name):
    try:
        model = torch.hub.load("intel-isl/MiDaS", models[model_name]).eval()
        print("Model found in cache.")
        return model.to(device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Downloading the model. This may take a moment...")
        model = torch.hub.load("intel-isl/MiDaS", models[model_name], force_reload=True).eval()
        return model.to(device)

# Downscale the image
def downscale_image(image, size):
    downscaled_image = image.resize(size)
    return downscaled_image

# Upscale the depth map
def upscale_depth_map(depth_map, target_size):
    upscaled_depth_map = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_LINEAR)
    return upscaled_depth_map

# Estimate depth from the input image
def estimate_depth(original_image, downscaled_size, selected_model):
    # Load the selected depth estimation model
    model = load_depth_model(selected_model)

    # Convert the image to RGB color space
    rgb_image = original_image.convert("RGB")

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_image.width / original_image.height

    # Log: Estimating depth
    print("Estimating depth...")

    # Squash the image to a square
    squared_image = rgb_image.resize((downscaled_size[0], int(downscaled_size[0] / aspect_ratio)))

    # Downscale the image 
    input_image = transforms.ToTensor()(squared_image).unsqueeze(0).to(device)

    # Perform depth estimation
    with torch.no_grad():
        depth_prediction = model(input_image)

    # Transfer the depth prediction to CPU and convert to numpy array
    depth_map = depth_prediction.squeeze().cpu().numpy()

    # Determine the target size for upscaling
    target_size = (original_image.width, original_image.height)

    # Upscale the depth map to the original image size
    upscaled_depth_map = upscale_depth_map(depth_map, target_size)

    # Normalize the depth values (0-1)
    upscaled_depth_map = (upscaled_depth_map - upscaled_depth_map.min()) / (
            upscaled_depth_map.max() - upscaled_depth_map.min())

    # Convert the depth map to an image
    upscaled_depth_map_image = Image.fromarray((upscaled_depth_map * 255).astype(np.uint8))

    # Log: Depth estimation complete
    print("Depth estimation complete")

    return upscaled_depth_map_image


# GUI layout
layout = [
    [sg.Image(key="-IMAGE-", size=(400, 400))],
    [sg.Button("Select Image/Video"), sg.Text("Model Selection"), sg.Combo(list(models.keys()), size=(12, 1), default_value=default_model, key="-MODEL-")],
    [sg.Button("Estimate Depth"), sg.Button("Export Depth Map"), sg.Button("Export as Video"), sg.Button("Exit")]
]

# Create the window
window = sg.Window("Depth Estimation", layout, size=(500, 600))

# Initialize attributes
original_image = None
depth_map_image = None
video_frames = []
video_file_path = ""

# Event loop
while True:
    event, values = window.read()

    # Handle window close or "Exit" button
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break

    # Handle the "Select Image/Video" button
    if event == "Select Image/Video":
        file_path = sg.popup_get_file("Select an image or video", file_types=(("Image Files", "*.jpg;*.jpeg;*.png"), ("Video Files", "*.mp4;*.mov")))
        if file_path:
            try:
                if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Selected file is an image
                    original_image = Image.open(file_path)
                    # Resize the image to fit the maximum size
                    max_width, max_height = 400, 400
                    resized_image = original_image.copy()
                    resized_image.thumbnail((max_width, max_height))
                    window["-IMAGE-"].update(data=ImageTk.PhotoImage(resized_image))
                    video_frames = []
                    video_file_path = ""
                elif file_path.lower().endswith((".mp4", ".mov")):
                    # Selected file is a video
                    video_file_path = file_path
                    video_frames = []
                    # Read the video frames
                    video_capture = cv2.VideoCapture(video_file_path)
                    while True:
                        ret, frame = video_capture.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_frames.append(Image.fromarray(frame))
                    video_capture.release()
                    if len(video_frames) > 0:
                        original_image = video_frames[0]
                        # Resize the image to fit the maximum size
                        max_width, max_height = 400, 400
                        resized_image = original_image.copy()
                        resized_image.thumbnail((max_width, max_height))
                        window["-IMAGE-"].update(data=ImageTk.PhotoImage(resized_image))
            except:
                sg.popup_error("Failed to open the image or video file.")

    # Handle the "Estimate Depth" button
    if event == "Estimate Depth":
        if original_image:
            try:
                selected_model = values["-MODEL-"]

                downscaled_size = default_downscaled_size
                # Calculate the aspect ratio of the original image
                aspect_ratio = original_image.width / original_image.height

                # Log: Starting depth estimation
                print("Starting depth estimation")
                print(f"Selected model: {selected_model}")
                print(f"Aspect ratio: {aspect_ratio}")
                print(f"Downscaled size: {downscaled_size}")

                if video_file_path:
                    # Estimate depth for each frame in the video
                    estimated_depth_frames = []
                    for frame in video_frames:
                        estimated_depth_frame = estimate_depth(frame, downscaled_size, selected_model)
                        estimated_depth_frames.append(estimated_depth_frame)

                    # Update video frames with depth maps
                    video_frames = estimated_depth_frames
                    depth_map_image = video_frames[0]
                else:
                    # Estimate depth for the single image
                    depth_map_image = estimate_depth(original_image, downscaled_size, selected_model)

                # Resize the depth map image to fit the maximum size
                resized_image = depth_map_image.copy()
                resized_image.thumbnail((max_width, max_height))
                window["-IMAGE-"].update(data=ImageTk.PhotoImage(resized_image))

                # Log: Depth estimation complete
                print("Depth estimation complete")
            except:
                sg.popup_error("Failed to estimate depth.")

    # Handle the "Export Depth Map" button
    if event == "Export Depth Map":
        if depth_map_image:
            try:
                save_path = sg.popup_get_file("Save depth map", save_as=True, file_types=(("PNG Image", "*.png"),))
                if save_path:
                    depth_map_image.save(save_path)
                    sg.popup("Depth map exported successfully!")
            except:
                sg.popup_error("Failed to export the depth map.")

    # Handle the "Export as Video" button
    if event == "Export as Video":
        if video_frames:
            try:
                save_path = sg.popup_get_file("Save video file", save_as=True, file_types=(("MP4 Video", "*.mp4"), ("MOV Video", "*.mov")))
                if save_path:
                    # Get the original image size
                    original_size = video_frames[0].size
                    # Create a video writer object
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(save_path, fourcc, 30, original_size)

                    # Write each frame to the video file
                    for frame in video_frames:
                        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                        video_writer.write(frame)

                    # Release the video writer
                    video_writer.release()

                    sg.popup("Video exported successfully!")
            except:
                sg.popup_error("Failed to export the video.")

# Close the window
window.close()
