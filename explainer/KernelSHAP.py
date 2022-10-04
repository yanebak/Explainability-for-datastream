import shap
import os
import matplotlib.pyplot as plt
import json
import numpy as np

feature_names1 = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature','Thermocouple', 'Voltage','Volume Flow RateRMS']
#array_feature_names = np.array(['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature','Thermocouple', 'Voltage','Volume Flow RateRMS'])


class KernelSHAP:
    """
        Thois class implements a customized version of kernel Explainer of SHAP
        Kernel SHAP is a variant of SHAP that uses LIME+ Shapley values
    """
    background_set =  None 

    def __init__(self, background):
        self.background = background
    

    def get_background_set(self, X_train, background_size=100, kmeans=False, random=False):
        """
        Get the first background_size records from x_train data and return it. Used for SHAP explanation process.
        Args:
            X_train (data frame): the data we will get the background set from
            background_size (int): The number of records to select from X_train. Default value is 100.
            kmeans(boolean)  : Is True when we use kmeans of shap for the background set.
            random(boolean) : Select randomly a background_size samples in X_train
        Returns:
            data frame: Records from x_train that will be the background set of the explanation of the record that we
                        explain at that moment using SHAP.
        """

        if kmeans:
            background_set = shap.kmeans(X_train,background_size)
        if random:
            background_set = shap.sample(X_train,background_size)

        background_set = X_train.head(background_size)
        return background_set
    
    def train_explainer(self,function_score,anormalous_instance_to_explain):
        # Define the type of background
        background_set = self.get_background_set(self.background)
        # print("$$$$$$$$$$$$",background_set.shape)
        #print("**************",background_set)
        explainer = shap.KernelExplainer(function_score, background_set, link = "identity")
        # We calculate now the shap values
        shap_values = explainer.shap_values(X = anormalous_instance_to_explain, nsamples = "auto")

        return shap_values, explainer.expected_value
    
    def save_plot(self, function_score, idx, data_without_y,anormalous_instance_to_explain, bar_plot=True, force_plot=True, json_file=True,dic_explanations={}):
        PATH =  os.getcwd()
        results_dir = os.path.join(PATH, 'Explanations/')
        results_dir1 = os.path.join(results_dir, 'Bar_plot/')
        results_dir2 = os.path.join(results_dir, 'Force_plot/')
        results_dir3 = os.path.join(results_dir, 'Json/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        if not os.path.isdir(results_dir1):
            os.makedirs(results_dir1)
        if not os.path.isdir(results_dir2):
            os.makedirs(results_dir2)
        if not os.path.isdir(results_dir3):
            os.makedirs(results_dir3)
        filename = 'Instance-'+str(data_without_y.iloc[idx,:].name)+'.png'
        shap_value_single = self.train_explainer(function_score,anormalous_instance_to_explain)[0]
        expected_value =  self.train_explainer(function_score,anormalous_instance_to_explain)[1]
        if json_file:
            array_feature_names = np.array(data_without_y.columns)
            feature_ordered = np.argsort(-np.abs(shap_value_single))
            features_important_for_instance= array_feature_names[feature_ordered]
            dic_explanations[str(data_without_y.iloc[idx,:].name)] = list(features_important_for_instance)[:10]
            with open(results_dir3+'explanation_shap.json', 'w') as jsonFile:
                json.dump(dic_explanations, jsonFile,indent=6)

        if bar_plot:
            plt_1 = plt.figure(figsize=(18,5))
            plt.title("Instance-", 
                fontdict={  'family': 'serif', 
                            'color' : 'darkblue',
                            'weight': 'bold',
                            'size': 17},
                 loc='center'
                 )

            shap.bar_plot(shap_value_single,feature_names=data_without_y.columns, show=False)
            plt.savefig(results_dir1 + filename)

        if force_plot:
            plt.title("Instance-", 
                fontdict={  'family': 'serif', 
                            'color' : 'darkblue',
                            'weight': 'bold',
                            'size': 17},
                 loc='center'
                 )
            shap.force_plot(expected_value,shap_value_single,anormalous_instance_to_explain,matplotlib=True,show=False)
            plt.savefig(results_dir2 + filename)
        
    
    

