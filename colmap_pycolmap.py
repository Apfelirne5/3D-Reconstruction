import gradio as gr
import os
import cv2
import pycolmap
import numpy as np
from pathlib import Path
from PIL import Image
import shutil


def extract_frames(video_path, output_folder, frame_interval=20):
    """Extracts frames from a video at a given interval."""
    print("Extracting frames from video...")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, frame_count = 0, 0
    success, image = cap.read()
    
    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
            print("frame_path: ", frame_path)
            cv2.imwrite(frame_path, image)
            frame_count += 1
        success, image = cap.read()
        count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames.")
    return f"Extracted {frame_count} frames."


def select_best_images(image_folder, max_images=50):
    """Splits images into `max_images` parts and selects the best one from each part based on sharpness."""
    image_paths = list(Path(image_folder).glob("*.jpg"))
    print("image_paths: ", image_paths[0])
    scores = {}
    
    if not image_paths:
        return "Error: No images found in the folder."

    # Compute sharpness score for each image using Laplacian variance
    for img_path in image_paths:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        scores[img_path] = laplacian_var
    
    # Sort images by sharpness (Laplacian variance) in descending order
    sorted_images = sorted(scores, key=scores.get, reverse=True)
    
    # Split the sorted images into `max_images` parts and select the best one from each part
    num_images = len(sorted_images)
    part_size = num_images // max_images
    selected_images = []

    for i in range(max_images):
        # Determine the range for this part
        start_idx = i * part_size
        end_idx = (i + 1) * part_size if i != max_images - 1 else num_images
        part_images = sorted_images[start_idx:end_idx]

        if part_images:
            # Select the best image from this part (first in the sorted list)
            selected_images.append(part_images[0])

    # Clear the images directory and move selected best images there
    #for img_path in Path(image_folder).glob("*.jpg"):
    #    os.remove(img_path)
    
    for img_path in selected_images:
        new_path = os.path.join(image_folder, os.path.basename(img_path))
        try:
            shutil.move(str(img_path), new_path)
        except FileNotFoundError:
            print(f"Error: Could not move {img_path} (file missing).")
    
    print(f"Selected {len(selected_images)} best images from each part.")

    return f"Selected {len(selected_images)} best images from each part."

def scale_images(workspace, scaling_option):
    """Scales images and saves them in a new folder."""
    workspace = os.path.abspath(workspace)

    # Check if workspace already ends with 'images'
    if not workspace.endswith("images"):
        images_folder = os.path.join(workspace, "images")
    else:
        images_folder = workspace  # If already ends with 'images', use workspace as it is
    
    # Modify original_folder to use 'images_original' if 'workspace' ends with 'images'
    if workspace.endswith("images"):
        original_folder = os.path.join(workspace[:-6], "images_original")  # Remove 'images' and add 'images_original'
    else:
        original_folder = os.path.join(workspace, "images_original")

    print("images_folder: ", images_folder)
    print("original_folder: ", original_folder)

    if not os.path.exists(images_folder):
        return f"Error: The images folder was not found at {images_folder}"

    if scaling_option == "No Scaling":
        return "No scaling selected. Using original images."

    if os.path.exists(original_folder):
        return f"Error: {original_folder} already exists. Please remove or rename it before scaling."

    os.rename(images_folder, original_folder)
    os.makedirs(images_folder, exist_ok=True)

    supported_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"]
    files_processed = 0
    errors = []

    for filename in os.listdir(original_folder):
        print("scale_images: ", scale_images)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        input_path = os.path.join(original_folder, filename)
        output_path = os.path.join(images_folder, filename)

        try:
            with Image.open(input_path) as img:
                if scaling_option == "Half":
                    new_width, new_height = img.width // 2, img.height // 2
                elif scaling_option == "Quarter":
                    new_width, new_height = img.width // 4, img.height // 4
                elif scaling_option == "Eighth":
                    new_width, new_height = img.width // 8, img.height // 8
                elif scaling_option == "1600k":
                    max_dim = max(img.width, img.height)
                    if max_dim > 1600:
                        scale = 1600 / max_dim
                        new_width, new_height = int(img.width * scale), int(img.height * scale)
                    else:
                        new_width, new_height = img.width, img.height
                else:
                    errors.append(f"Unsupported scaling option: {scaling_option}")
                    continue

                resample = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                img_resized = img.resize((new_width, new_height), resample)
                img_resized.save(output_path)
                files_processed += 1

        except Exception as e:
            errors.append(f"Error processing {filename}: {str(e)}")

    result = f"Processed {files_processed} images."
    if errors:
        result += "\nEncountered errors:\n" + "\n".join(errors)
    result += f"\nNew images in: {images_folder}\nOriginal images in: {original_folder}"
    return result


def run_colmap(workspace, matching_type):
    """Runs COLMAP reconstruction using pycolmap."""
    workspace = os.path.abspath(workspace)
    images_folder = os.path.join(workspace, "images")
    # Check if workspace already ends with 'images'
    if not workspace.endswith("images"):
        images_folder = os.path.join(workspace, "images")
    else:
        images_folder = workspace  # If already ends with 'images', use workspace as it is

    # Modify original_folder to use 'images_original' if 'workspace' ends with 'images'
    if workspace.endswith("images"):
        original_folder = os.path.join(workspace[:-6], "images_original")  # Remove 'images' and add 'images_original'
    else:
        original_folder = os.path.join(workspace, "images_original")

    # Modify original_folder to use 'images_original' if 'workspace' ends with 'images'
    if workspace.endswith("images"):
        db_path = os.path.join(workspace[:-6], "database.db")  # Remove 'images' and add 'images_original'
        sparse_path = os.path.join(workspace[:-6], "sparse")
    else:
        db_path = os.path.join(workspace, "database.db")
        sparse_path = os.path.join(workspace, "sparse")
        
    log = []

    print("images_folder: ", images_folder)

    def add_log(msg):
        print(msg)
        log.append(msg)

    add_log(f"Workspace: {workspace}")

    if not os.path.exists(images_folder):
        add_log(f"Error: Images folder not found at {images_folder}")
        return "\n".join(log)

    os.makedirs(sparse_path, exist_ok=True)

    try:
        add_log("=== Feature Extraction ===")
        pycolmap.extract_features(db_path, images_folder)
        add_log("Feature extraction completed.")
    except Exception as e:
        add_log(f"Feature extraction failed: {str(e)}")
        return "\n".join(log)

    try:
        add_log(f"=== {matching_type} Matching ===")
        if matching_type == "Exhaustive":
            pycolmap.match_exhaustive(db_path)
        elif matching_type == "Sequential":
            pycolmap.match_sequential(db_path)
        elif matching_type == "Spatial":
            pycolmap.match_spatial(db_path)
        else:
            add_log(f"Invalid matching type: {matching_type}")
            return "\n".join(log)
        add_log("Feature matching completed.")
    except Exception as e:
        add_log(f"Feature matching failed: {str(e)}")
        return "\n".join(log)

    try:
        add_log("=== Sparse Reconstruction ===")
        maps = pycolmap.incremental_mapping(db_path, images_folder, sparse_path)
        if not maps:
            add_log("Reconstruction failed: No maps generated.")
            return "\n".join(log)
        
        recon_dir = os.path.join(sparse_path, "0")
        os.makedirs(recon_dir, exist_ok=True)
        maps[0].write(recon_dir)
        add_log("Sparse reconstruction completed.")
    except Exception as e:
        add_log(f"Sparse reconstruction failed: {str(e)}")
        return "\n".join(log)

    # Verify output files
    result_dir = os.path.join(sparse_path, "0")
    expected_files = [os.path.join(result_dir, fname) for fname in ["cameras.bin", "images.bin", "points3D.bin"]]

    if all(os.path.exists(f) for f in expected_files):
        add_log("COLMAP reconstruction succeeded!")
        add_log(f"Results in: {result_dir}")
    else:
        add_log("Error: Missing output files.")
        for f in expected_files:
            add_log(f"{f}: {'Found' if os.path.exists(f) else 'Missing'}")

    return "\n".join(log)


def process_workflow(workspace, scaling_option, matching_type, max_images):
    """Orchestrates image extraction (if video), scaling, and COLMAP reconstruction."""
    logs = ["=== Image Preparation ==="]
    
    workspace_path = Path(workspace)
    
    # If the input is a video file
    if workspace_path.is_file() and workspace_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        video_folder = workspace_path.parent / "images"
        logs.append(extract_frames(str(workspace_path), str(video_folder)))
        
        # Ensure max_images is a valid integer
        try:
            max_images = int(max_images)
            if max_images <= 0:
                raise ValueError
        except ValueError:
            return "Error: Max images must be a positive integer."
        
        logs.append(select_best_images(str(video_folder), max_images))
        workspace = str(video_folder)  # Update workspace to the images folder
    
    # Image scaling
    scale_result = scale_images(workspace, scaling_option)
    logs.append(scale_result)
    
    # COLMAP reconstruction
    logs.append("\n=== COLMAP Reconstruction ===")
    colmap_result = run_colmap(workspace, matching_type)
    logs.append(colmap_result)
    
    return "\n".join(logs)


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# COLMAP Workflow with PyCOLMAP")
    gr.Markdown(
        "Provide a workspace directory containing an 'images' folder or a video file. "
        "If a video is provided, frames are extracted first, then processed with COLMAP."
    )
    
    workspace_input = gr.Textbox(label="Workspace Directory or Video File", placeholder="/path/to/workspace_or_video.mp4")
    scaling_input = gr.Radio(
        choices=["No Scaling", "Half", "Quarter", "Eighth", "1600k"],
        label="Image Scaling",
        value="No Scaling"
    )
    matching_input = gr.Radio(
        choices=["Exhaustive", "Sequential", "Spatial"],
        label="Feature Matching",
        value="Exhaustive"
    )
    max_images_input = gr.Textbox(label="Max Best Images to Extract (if using video)", placeholder="50", value="50")
    
    run_button = gr.Button("Run")
    output_log = gr.Textbox(label="Output Log", lines=25)

    run_button.click(
        fn=process_workflow,
        inputs=[workspace_input, scaling_input, matching_input, max_images_input],
        outputs=output_log
    )

demo.launch()