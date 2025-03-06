# mri-to-pet-image-synthesis
## Overview
This project aims to develop a deep learning model to synthesize PET images from MRI scans. The goal is to provide a non-invasive and cost-effective method for generating PET-like images, which can be useful in various medical diagnoses and research.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/aaditya0106/mri-to-pet-image-synthesis.git
    ```
2. Navigate to the project directory:
    ```sh
    cd mri-to-pet-image-synthesis
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your MRI data and place it in the `../t1_flair_asl_fdg_preprocessed/` directory.

2. Train the model:
    ```sh
    python train.py
    ```
3. Generate PET images from MRI scans:
    ```sh
    python generate.py
    ```