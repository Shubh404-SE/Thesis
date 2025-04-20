import os
import numpy as np
import cv2
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import logging
from typing import Optional, Tuple, List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

class DICOMLoader:
    """Handles DICOM series loading and validation."""
    
    @staticmethod
    def load_dicom_series(dicom_folder: str) -> Optional[sitk.Image]:
        """Loads a DICOM series into a 3D volume using SimpleITK.
        
        Args:
            dicom_folder: Path to the folder containing DICOM files
            
        Returns:
            SimpleITK image object or None if loading fails
        """
        if not os.path.exists(dicom_folder):
            logging.warning(f"DICOM folder not found: {dicom_folder}")
            return None
            
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        
        if not dicom_names:
            logging.warning(f"No DICOM files found in: {dicom_folder}")
            return None
            
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
            logging.info(f"Successfully loaded DICOM series from {dicom_folder}")
            return image
        except Exception as e:
            logging.error(f"Error loading DICOM series from {dicom_folder}: {e}")
            return None

class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def skull_strip(sitk_img: sitk.Image) -> sitk.Image:
        """Applies skull stripping to the input image.
        
        Args:
            sitk_img: Input SimpleITK image
            
        Returns:
            Skull-stripped SimpleITK image
        """
        img_data = sitk.GetArrayFromImage(sitk_img)
        
        # Apply Gaussian smoothing
        smoothed_data = gaussian_filter(img_data, sigma=1)

        # Compute Otsu's threshold
        threshold = threshold_otsu(smoothed_data)
        brain_mask = smoothed_data > (threshold * 0.8)

        # Fill holes in the mask and remove small objects
        brain_mask = binary_fill_holes(brain_mask)
        brain_mask = remove_small_objects(brain_mask, min_size=200)

        # Apply the mask to the image
        stripped_img = smoothed_data * brain_mask
        
        # Convert back to SimpleITK image
        stripped_sitk = sitk.GetImageFromArray(stripped_img)
        stripped_sitk.CopyInformation(sitk_img)
        return stripped_sitk

    @staticmethod
    def normalize(sitk_img: sitk.Image) -> sitk.Image:
        """Normalizes the intensity while preserving contrast.
        
        Args:
            sitk_img: Input SimpleITK image
            
        Returns:
            Normalized SimpleITK image
        """
        img_data = sitk.GetArrayFromImage(sitk_img)
        
        min_val, max_val = np.percentile(img_data, (1, 99))  # Exclude extreme outliers
    
        if max_val - min_val == 0:
            norm_data = img_data
        else:
            norm_data = np.clip((img_data - min_val) / (max_val - min_val + 1e-8), 0, 1)

        norm_sitk = sitk.GetImageFromArray(norm_data)
        norm_sitk.CopyInformation(sitk_img)
        return norm_sitk

    @staticmethod
    def save_preprocessed_image(preprocessed_data: sitk.Image, save_path: str) -> None:
        """Saves the most representative slice as PNG.
        
        Args:
            preprocessed_data: Processed SimpleITK image
            save_path: Path to save the output PNG
        """
        img_data = sitk.GetArrayFromImage(preprocessed_data)
        middle_index = img_data.shape[0] // 2
        final_img = (img_data[middle_index] * 255).astype(np.uint8)
        cv2.imwrite(save_path, final_img)

class DatasetProcessor:
    """Handles dataset processing and management."""
    
    def __init__(self, dataset_path: str, output_path: str, max_patients: int = 250):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.max_patients = max_patients
        self.mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.dicom_loader = DICOMLoader()
        self.image_processor = ImageProcessor()
        
    def process_patient(self, patient_path: str, output_patient_path: str) -> bool:
        """Processes all MRI types for a single patient.
        
        Args:
            patient_path: Path to patient's DICOM data
            output_patient_path: Path to save processed data
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            os.makedirs(output_patient_path, exist_ok=True)
            
            for mri_type in self.mri_types:
                input_mri_path = os.path.join(patient_path, mri_type)
                output_mri_path = os.path.join(output_patient_path, mri_type)
                os.makedirs(output_mri_path, exist_ok=True)

                # Load DICOM series
                dicom_img = self.dicom_loader.load_dicom_series(input_mri_path)
                if dicom_img is None:
                    continue

                # Process the image
                skull_stripped = self.image_processor.skull_strip(dicom_img)
                normalized = self.image_processor.normalize(skull_stripped)

                # Save the processed image
                final_image_path = os.path.join(output_mri_path, "preprocessed.png")
                self.image_processor.save_preprocessed_image(normalized, final_image_path)
            
            return True
        except Exception as e:
            logging.error(f"Error processing patient {patient_path}: {e}")
            return False

    def run_preprocessing(self) -> None:
        """Runs preprocessing on the dataset, limited to max_patients."""
        patient_folders = sorted(os.listdir(self.dataset_path))[:self.max_patients]
        
        for patient in tqdm(patient_folders, desc="Processing patients"):
            patient_path = os.path.join(self.dataset_path, patient)
            output_patient_path = os.path.join(self.output_path, patient)
            
            if self.process_patient(patient_path, output_patient_path):
                logging.info(f"Successfully processed patient {patient}")
            else:
                logging.warning(f"Failed to process patient {patient}")

        logging.info(f"Preprocessing completed for {len(patient_folders)} patients! 🚀")

if __name__ == "__main__":
    # Example usage
    DATASET_PATH = "./brats2021"  # Path to DICOM dataset
    OUTPUT_PATH = "./output"  # Output directory
    
    processor = DatasetProcessor(DATASET_PATH, OUTPUT_PATH, max_patients=250)
    processor.run_preprocessing() 