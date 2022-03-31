# MMF-DSS
This repository hosts all code utilized in the creation of the manuscript entitled "Toward Automated Instructor Pilots in Legacy Air Force Systems: Physiology-based Flight Difficulty Classification via Machine Learning" by Caballero, Gaw, Jenkins and Johnstone. Namely, the code provided herein allows the interested reader to develop, train and test the Multi-Modal Functional-based Decision Support System (MMF-DSS) developed in the aforementioned manuscript. 

## Obtaining the Data
The data leveraged within the study is the property of the USAF-MIT AI Accelerator. For access to this data, please submit inquiries at https://aia.mit.edu/contact-us/. After submitting an inquriy, the owning office will contact you with further guidance and instructions. 

## Summarizing the Pipeline's Implementation

Multiple coding languages are leveraged within the MMF-DSS based on myriad factors. As discussed in the manuscript's text, the MMF-DS incorporates FPCA and summary statistic methods for feature generation and BorutaSHAP for feature selection. Whereas Python packages are available for PCA (e.g., FDApy), they are less developed than their MATLAB counterparts, especially with regards to the handling of sparse data sets. As such, the PACE package from MATLAB is utilized in the MMF-DSS pipeline. Alternatively, BorutaSHAP (as well as its TreeSHAP dependency) are not yet developed in MATLAB; as such, Python tools are utilized for feature selection. Whereas either MATLAB or Python could have been utilized for the classier's development, MMF-DSS leverages MATLAB due to the designer's preference.

Therefore, to recreate the results presented by Caballero et al, the interested researcher should (1) obtain permission from the USAF-MIT AI Accelerator to access the CogPilot data set, (2) run the files within the "Feature Generation" folder in MATLAB, (3) utilize the output .csv within the Python code provided in the "Feature Selection" folder, and (4) import the output of the BorutaSHAP algorithm into the MATLAB code provisioned in "Classifer Development" folder. 

