import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Ensures Matplotlib uses a GUI backend for displaying images

def find_dicom_files(root_path):
    """ Recursively find all DICOM files in the dataset directory. """
    dicom_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('.dcm'):  # Adjust this if your files have a different extension
                dicom_files.append(os.path.join(dirpath, file))
    return dicom_files

def load_dicom_image(dicom_path):
    """ Load a DICOM file and return its data. """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        return dicom_data
    except Exception as e:
        print(f"Error loading DICOM file {dicom_path}: {e}")
        return None

def visualize_dicom_image(dicom_data):
    """ Display the DICOM image if it contains pixel data. """
    if dicom_data:
        if hasattr(dicom_data, "pixel_array"):
            print("✅ DICOM file contains an image. Saving...")
            plt.imshow(dicom_data.pixel_array, cmap=plt.cm.bone)
            plt.title("CT Image")
            plt.axis('off')  # Hide axes
            plt.savefig('dicom_image.png', bbox_inches='tight')  # Save the image
            print("Image saved as dicom_image.png")
        else:
            print("⚠️ DICOM file does not contain an image (no pixel_array).")
    else:
        print("❌ No image to display.")

if __name__ == "__main__":
    # Root path of your dataset
    dataset_path = "/fab3/btech/2022/sreejita.das22b/Attention-Gated-Networks/Pancreas_Small1"

    # Find all DICOM files
    dicom_files = find_dicom_files(dataset_path)
    print(f"🔍 Found {len(dicom_files)} DICOM files.")

    if dicom_files:
        # Try loading and displaying the first valid DICOM file
        for dicom_path in dicom_files:
            print(f"Loading: {dicom_path}")
            dicom_data = load_dicom_image(dicom_path)

            # Print metadata for debugging
            print(dicom_data)

            if dicom_data and hasattr(dicom_data, "pixel_array"):
                visualize_dicom_image(dicom_data)
                break  # Stop after displaying the first valid image
    else:
        print("❌ No DICOM files found in the dataset directory.")
