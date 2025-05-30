import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import os
import numpy as np
import logging
from tqdm import tqdm  # for progress bar

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_filelist(root_path):
    """Get list of patient directories containing both data.nii and mask.nii"""
    filelist = []
    patient_numbers = []
    
    for root, dirs, files in os.walk(root_path):
        for dirname in dirs:
            patient_dir = os.path.join(root, dirname)
            data_path = os.path.join(patient_dir, "data.nii")
            mask_path = os.path.join(patient_dir, "mask.nii")
            
            # Check if both required files exist
            if os.path.exists(data_path) and os.path.exists(mask_path):
                filelist.append(patient_dir)
                # Extract patient number from path (adjust indices based on your path structure)
                patient_number = patient_dir.split('\\')[-1][:7]  # Adjust this based on your path structure，需要自行修稿
                patient_numbers.append(patient_number)
    
    return filelist, patient_numbers

def excel_one_line_to_list(path, col):
    """Read single column from Excel file"""
    try:
        df = pd.read_excel(path, usecols=[col], names=None)
        return [item[0] for item in df.values.tolist()]
    except Exception as e:
        logger.error(f"Error reading Excel file {path}: {e}")
        raise

def load_group_data():
    """Load group information from Excel files"""
    try:
        # Load group A data
        A_num = excel_one_line_to_list("C:\\Users\\LL\\Desktop\\second_A_group.xlsx", 1)
        B_num = excel_one_line_to_list("C:\\Users\\LL\\Desktop\\second_B_group.xlsx", 1)
        A_sex = excel_one_line_to_list("C:\\Users\\LL\\Desktop\\second_A_group.xlsx", 4)
        B_sex = excel_one_line_to_list("C:\\Users\\LL\\Desktop\\second_B_group.xlsx", 4)
        A_age = excel_one_line_to_list("C:\\Users\\LL\\Desktop\\second_A_group.xlsx", 7)
        B_age = excel_one_line_to_list("C:\\Users\\LL\\Desktop\\second_B_group.xlsx", 7)#sex和age等临床特征可以通过这部分代码添加，也可以之后在数据融合部分手动添加
        
        # Create dictionaries
        num_sex = {**dict(zip(A_num, A_sex)), **dict(zip(B_num, B_sex))}
        num_age = {**dict(zip(A_num, A_age)), **dict(zip(B_num, B_age))}
        
        return A_num, B_num, num_sex, num_age
    except Exception as e:
        logger.error(f"Error loading group data: {e}")
        raise

def main():
    try:
        # Initialize paths and parameters
        data_path = 'D:\\A'
        output_path = 'D:\\A\\测试数据特征.xlsx'
        
        # Configure feature extractor
        extractor_params = {
            'binWidth': 25,
            'resampledPixelSpacing': None,
            'interpolator': sitk.sitkBSpline,
            'enableCExtensions': True
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_params)
        
        # Get file list and patient info
        filelist, patient_numbers = get_filelist(data_path)
        logger.info(f"Found {len(filelist)} valid patient directories")
        
        # Load group information
        A_num, B_num, num_sex, num_age = load_group_data()
        
        # Prepare results dataframe
        results = []
        failed_cases = []
        
        # Process each patient
        for i, (patient_dir, patient_num) in tqdm(enumerate(zip(filelist, patient_numbers)), 
                                        total=len(filelist), desc="Processing patients"):
            try:
                image_path = os.path.join(patient_dir, "data.nii")
                mask_path = os.path.join(patient_dir, "mask.nii")
                
                logger.info(f"Processing patient {patient_num} ({i+1}/{len(filelist)})")
                
                # Extract features
                feature_vector = extractor.execute(image_path, mask_path)
                
                # Add patient metadata
                feature_vector['Patient_number'] = patient_num
                
                # Add group label
                if int(patient_num) in A_num:
                    feature_vector['label'] = '1'
                    logger.debug(f"Patient {patient_num} in group A")
                elif int(patient_num) in B_num:
                    feature_vector['label'] = '0'
                    logger.debug(f"Patient {patient_num} in group B")
                
                # Add to results
                results.append(feature_vector)
                logger.info(f"Successfully processed patient {patient_num}")
                
            except Exception as e:
                logger.error(f"Failed to process patient {patient_num}: {e}")
                failed_cases.append({
                    'patient': patient_num,
                    'error': str(e)
                })
                continue
        
        # Convert results to DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Save results
            df.to_excel(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Save failed cases if any
            if failed_cases:
                failed_df = pd.DataFrame(failed_cases)
                failed_output = os.path.join(os.path.dirname(output_path), 'failed_cases.xlsx')
                failed_df.to_excel(failed_output, index=False)
                logger.warning(f"{len(failed_cases)} cases failed. Details saved to {failed_output}")
        else:
            logger.error("No patients were successfully processed")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()