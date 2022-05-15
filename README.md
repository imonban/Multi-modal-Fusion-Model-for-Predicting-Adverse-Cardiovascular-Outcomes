# Multi-modal-Fusion-Model-for-Predicting-Adverse-Cardiovascular-Outcomes

The code for the multi-modal fusion model for prediction of adverse cardiovascular outcomes using both ECG and EHR Data.
The code was implemented using Python 3.8 and Tensorflow 2.5.0 
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)


## Model Description

The model has been built for prediction of cardiovascular outcomes such as Stroke, Chronic Heart Failure and Mortality using the ECG data as well as the Electronic Health Record(EHR) data. 

**The ECG Data was extracted from the images and preprocessed before being sent to the model**

### ECG Image 
![A10823](https://user-images.githubusercontent.com/44440114/168468978-e74bd558-d51e-4b60-9142-9cce4e89c89a.png)

### Preprocessed ECG Image 
![A10823_plot](https://user-images.githubusercontent.com/44440114/168468998-2713d966-4cff-4335-9d5a-18f9687c86d6.png)

## Model
![ModelImage](https://user-images.githubusercontent.com/44440114/168468895-6380a1b8-44ac-45ac-abef-3728e0b34add.png)

## How to Implement? 

Initially create an environment and install the requirements using the code below. 
```
pip install -r /path/to/requirements.txt
```

Then run the **main.py** file, giving which takes the input a CSV file containing EHR data, using the code present below.
**The EHR data was named as "Share_data_outcome_deidentified.csv" in our case**

```
!python3 main.py -i Share_data_outcome_deidentified.csv
```
This will automatically save the ROC.png file having the ROC scores

## Result
![ROC](https://user-images.githubusercontent.com/44440114/168469517-5bacce9b-6d1b-46a1-a579-58c6e19074c4.png)


