# Multi-modal-Fusion-Model-for-Predicting-Adverse-Cardiovascular-Outcomes

The code for the multi-modal fusion model for prediction of adverse cardiovascular outcomes using both ECG and EHR Data.
The code was implemented using 
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow) and [![TensorFlow 2.8](https://img.shields.io/badge/TensorFlow-2.8-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0) 

## Model Description

The model has been built for prediction of cardiovascular outcomes such as Stroke, Chronic Heart Failure and Mortality using the ECG data as well as the Electronic Health Record(EHR) data. 

**The ECG Data was extracted from the images and preprocessed before being sent to the model**

### ECG Image 
![A10823](https://user-images.githubusercontent.com/44440114/168468978-e74bd558-d51e-4b60-9142-9cce4e89c89a.png)

### Preprocessed ECG Image 
![A10823_plot](https://user-images.githubusercontent.com/44440114/168468998-2713d966-4cff-4335-9d5a-18f9687c86d6.png)

## Model
![ModelImage](https://user-images.githubusercontent.com/44440114/168468895-6380a1b8-44ac-45ac-abef-3728e0b34add.png)

## Files Present
1. **utility.py** - Contain the metrics 
2. **train.py**   - The code for building the required model and perform the training operation
3. **requirements.txt** - The libraries required to run the code
4. **plotsave.py** - Plots the ROC-AUC curve
5. **main.py**    - Running the code and saving the model results 
6. **ROC.png**    - Sample ROC curve generated 
7. **dataread.py**- Reading the ECG data, the EHR data, and the labels
8. **config.json**- Used in building the branch corresponding to the ECG data and controlling the number of convolution layers
9. **ModelImage.png** - Picture representation of our proposed fusion model

## How to Implement? 

Initially create an environment and install the requirements using the code below. 
```python
pip install -r /path/to/requirements.txt
```

Then run the **main.py** file, giving which takes the input a CSV file containing EHR data, using the code present below.
**The EHR data was named as "Share_data_outcome_deidentified.csv" in our case**

```python
!python3 main.py -i Share_data_outcome_deidentified.csv
```
This will automatically save the ROC.png file having the ROC scores
**The data we worked on has not been uploaded yet.**

## Result
![ROC](https://user-images.githubusercontent.com/44440114/168469517-5bacce9b-6d1b-46a1-a579-58c6e19074c4.png)

## Contribution
The code contribution of the Fusion Model was made [Amartya Bhattacharya](amartyacodes.github.io) while working as a research assistant at [Banerjee Lab, Arizona State University](https://labs.engineering.asu.edu/banerjeelab/) under the guidance of [Dr. Imon Banerjee](https://labs.engineering.asu.edu/banerjeelab/person/imon-banerjee/) and [Dr. Arindam Sanyal](https://labs.engineering.asu.edu/mixedsignals/).

I wish to convey my sincere gratitude to the professors for giving me the opportunity to work on this project.
