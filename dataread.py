import pandas as pd
import pickle
import numpy as np

def demo_process(df, column_name):
    df = df.drop(['primcauseofdeath','causeofdeath'], axis=1)
    k1 = df.shape[0]
    df = df[df[column_name]!= -1]
    k2 = df.shape[0] 
    print("Number of Rows containing NaN values: ", k1-k2)
    return df
    


def all_leads_process(demodf, column_name):

    """
    
    ind_var: an empty array which will contain the independent variables
    labels: an empty array which will contain the corresponding labels 
    column_name: a string containing the column information for storing the labels
    lead_name: a string containing the information of the lead that's going to be used
    image_type = a string mentioning type of Image 'PRE' or 'POST' -- will be added later
    
    
    """
    
    features = list(pd.read_csv('InputFeatures.csv')['Features'])

    ind_var = []
    labels = []
    demo = []
    image_name = []
    demodf = demo_process(demodf, column_name)

    demodf2 = demodf.drop(['PRE','POST', column_name],axis=1)
    with open('/mnt/storage/EGC_outcome/code_amartya/Documents/POST_image_info.pkl', 'rb') as f:
        image_info = pickle.load(f)
    image_info = pd.DataFrame(image_info)
  

    #Storing the independent variables and their corresponding labels
    
    for i in range(0,image_info.shape[0]):
        #Check if Image exists
        if str(image_info['Image Name'].iloc[i]).split('.')[1] == 'PNG':
        #Check if information for a given image exists

            if demodf.loc[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']].shape[0] > 0 and demodf.loc[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']].shape[0]>0:



                new_arr = []
                new_arr.append(np.array(image_info['Lead1'].iloc[i]))
                new_arr.append(np.array(image_info['Lead2'].iloc[i]))
                new_arr.append(np.array(image_info['Lead3'].iloc[i]))
                new_arr.append(np.array(image_info['Lead4'].iloc[i]))
                new_arr.append(np.array(image_info['Lead5'].iloc[i]))
                new_arr.append(np.array(image_info['Lead6'].iloc[i]))
                new_arr= np.array(new_arr)  
                
                
                image_name.append( image_info['Image Name'].iloc[i].replace('.PNG',''))
                ind_var.append(new_arr)
                labels.append(demodf.loc[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']][column_name].iloc[0])
                temp = demodf[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']][features].iloc[0].values
                demo.append(temp)             
        elif str(image_info['Image Name'].iloc[i]).split('.')[1] == 'png':

            if demodf.loc[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']].shape[0] > 0 and demodf.loc[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']].shape[0]>0:
           
                new_arr = []       
                new_arr.append(np.array(image_info['Lead1'].iloc[i]))
                new_arr.append(np.array(image_info['Lead2'].iloc[i]))
                new_arr.append(np.array(image_info['Lead3'].iloc[i]))
                new_arr.append(np.array(image_info['Lead4'].iloc[i]))
                new_arr.append(np.array(image_info['Lead5'].iloc[i]))
                new_arr.append(np.array(image_info['Lead6'].iloc[i]))
                new_arr= np.array(new_arr)
                ind_var.append(new_arr)
                temp = demodf[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']][features].iloc[0].values
                demo.append(temp)
                image_name.append( image_info['Image Name'].iloc[i].replace('.png',''))
                labels.append(demodf.loc[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']][column_name].iloc[0])
    ind_var = np.array(ind_var)
    ind_var = ind_var.reshape(ind_var.shape[0], ind_var.shape[2], 6)
    labels = np.array(labels)
    demo = np.array(demo).astype('float32')
    return ind_var, demo, labels,image_name

