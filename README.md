

# Small-Angle X-ray Scattering (SAXS) Data Analysis Based on Peak Fitting Methods

This project is the dissertation submitted for the degree of MSc Scientific and Data Intensive Computing, UCL. The target is to develop a data-analysis program focusing on fitting with time-resolved SAXS data. The program provides 3 common bell-like models for the fitting of curve including Gaussian, Lortzian and Pseudo-Voigt models. Additionally, it allows for the fitting of multiple models. But for now, it only accepts CSV and DAT files, and also ASCII files (XY format). 




## Running Suggestions
The program was developed with Python 3.9 on Visual Studio Code (VS Code). It is suggested to load the whole folder with VS Code or Pycharm. 


#### Before Running
Input the following commands in the terminal:
```sh
git clone https://github.com/ssiyuan/FindPeaks.git 
pip3 install scipy
pip3 install lmfit

```

#### **To run**
1. Select the file named 'main_CLI.py' and run it without any other parameters. Instructions will be shown in the terminal, just follow them. 
2. To terminate the running of program, use “Ctrl + C”. Otherwise, the program won't stop except the case that an error is caught.

Below shows the operating process and examples of input:
![image](https://github.com/ssiyuan/FindPeaks/blob/main/readme_images/CLI.png)

In the fitting progress, the initial guess for peaks center are very important. Take the 5-peak fitting in range [1.4, 6] for example, by observing 2d fugure of input data, guess for peak center can be set as 5.9, 5.3, 4.5, 3.9 and 2.3. In the terminal, the progress for this step should be like:
![image](https://github.com/ssiyuan/FindPeaks/blob/main/readme_images/input_example_ascii.png)

In this way an accurate fitting result can be generated (shown below). 
![image](https://github.com/ssiyuan/FindPeaks/blob/main/readme_images/5_peak_example.png)


## File Tree
```

filetree 
.
|____readme_images
| |____Flow_Chart_for_User.png
| |____CLI.png
|____import_files.py
|____test_data
| |____ASCII_data
| | |____1_1Pt_CeO2_cycle-00004.gr
| | |____1_1Pt_CeO2_cycle-00000.gr
| | |____1_1Pt_CeO2_cycle-00010.gr
| | |____1_1Pt_CeO2_cycle-00001.gr
| | |____1_1Pt_CeO2_cycle-00005.gr
| | |____1_1Pt_CeO2_cycle-00002.gr
| | |____1_1Pt_CeO2_cycle-00006.gr
| | |____1_1Pt_CeO2_cycle-00007.gr
| | |____1_1Pt_CeO2_cycle-00003.gr
| | |____1_1Pt_CeO2_cycle-00008.gr
| | |____1_1Pt_CeO2_cycle-00009.gr
| |____NH4OH-FAU-Practice-data.csv
|____README.md
|____fitting_peaks.py
|____main_CLI.py
|____Zeolite
| |____cbv720.dat
| |____cbv712.dat
| |____cbv901.csv
| |____cbv760.dat
|____main_evaluation.py
|____validation.py

```
This directory include codes, and data files used to test the performance:

1. The folder readme_images include figures used to write the README.md.
2. For the folder 'test_data', data used for the initial design of the program was stored. The ASCII files in folder named 'ASCII_data' proved that ASCII are accepted input type. 
3. The four files stored in folder 'Zeolite' refers to SAXS data with Si/Al differnt Ratios. Applied to study the mesopore formation with the file 'main_evaluation.py'.
4. There are two main functions in this folder tree. 
    3.1 The main() function in main_CLI.py was used to provide a friendly CLI for users, which can help them run the program without understanding codes. It has more functions than main_evaluation.py, including comparing three models to choose the most suitable one. 
    3.2 main() function in main_evaluation.py was used for the analysis of Mesopore structure formation from microporous zeolite. 
5. import_files.py includes functions needed to read data from DAT, CSV and some XY files. Also process the input data to transfer $2\theta$ to $q$ and change $y$ to $log(y)$ to make peaks more obvious.
6. fitting_peaks.py includes all functions used for curve fitting, including baseline detection, capturing data within peak range， and also the function of fitting, etc.
7. validation.py is used to check input to significant functions.



## Version Control

Version of this project was managed with Git. The current version is available in the GitHub repository.


## Author

ucapsqi@ucl.ac.uk


## Acknowledgements

Thank Prof Sanakar for his patient guidance and encouragement. In addtion, thank Junwen Gu for his encouragement and providing SAXS experiment data.

