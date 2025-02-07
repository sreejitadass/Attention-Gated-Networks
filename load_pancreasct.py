import os
import pydicom
import matplotlib.pyplot as plt

def load_dicom_image(dataset_path, filename):
    dicom_file = os.path.join(dataset_path, filename)
    try:
        dicom_data = pydicom.dcmread(dicom_file)
        return dicom_data
    except Exception as e:
        print(f"Error loading DICOM file {filename}: {e}")
        return None

def visualize_dicom_image(dicom_data):
    if dicom_data:
        plt.imshow(dicom_data.pixel_array, cmap=plt.cm.bone)
        plt.title("CT Image")
        plt.show()
    else:
        print("No image to display.")

if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "/fab3/btech/2022/sreejita.das22b/Attention-Gated-Networks/dataio/loader/Pancreas-CT-20200910.tcia"

    
    # List all files in the dataset directory
    try:
        files = os.listdir(dataset_path)
        dicom_files = [file for file in files if file.endswith('.dcm')]  # Adjust if the files have another extension
        
        if dicom_files:
            print(f"Found {len(dicom_files)} DICOM files.")
            # Load and visualize the first DICOM image
            dicom_data = load_dicom_image(dataset_path, dicom_files[0])
            visualize_dicom_image(dicom_data)
        else:
            print("No DICOM files found in the specified directory.")
    except Exception as e:
        print(f"Error accessing the dataset: {e}")
