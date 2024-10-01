# Topic:

41. Study the variational autoencoder GAN for medical image generation. Present a concrete example case study.

# Group member:

1. Nguyễn Thái Bình - 22BI13059
2. Nguyễn Minh Đức - 22BI13092
3. Vũ Tuấn Hải - 22BI13149
4. Cấn Minh Hiển - 22BI13154
5. Nguyễn Quang Hưng - 22BI13184
6. Nguyễn Thế Khải - 22BI13201

# Dataset

You can access the dataset, required packages, and the saved model file using the following link:  
[Download Dataset and Model Files](https://drive.google.com/drive/folders/1k0S2kniYhU5VCmShYn4V_zuRZTvHfNKH?usp=sharing)

## Instructions

Follow these steps to set up the environment and run the project:

1. **Install Anaconda:**
   - Download and install the `environment.yml` to have the requiment Anaconda packages

2. **Download the Brain Dataset:**
   - Download the brain dataset from the provided link.
   - Place the dataset inside the `vae_gan` directory. Ensure the folder structure looks like this:
     ```
     /vae_gan/
     ├── brain_dataset/
     │   ├── glioma/
     │   ├── pituitary/
     │   ├── notumor/
     │   └── meningioma/
     ```

3. **Download the Model File:**
   - Download the `vae_gan_model.pth` file from the link.
   - Place the model file in the `vae_gan` directory.

4. **Run the Project:**
   - Open your terminal or command prompt.
   - Navigate to the `vae_gan` directory.
   - Execute the following command to run the project:
     ```bash
     python main.py
     ```

Make sure to verify that all files are in their correct locations before running the script. If you encounter any issues, please check the folder structure and file paths.
