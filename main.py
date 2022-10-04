from numpy import sqrt, abs, round
from rrcf import rrcf
from sklearn.utils import shuffle
from pysad.utils import ArrayStreamer
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.models.integrations import ReferenceWindowModel
from pysad import models
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from  model.RobustRandomCutForest import RobustRandomCutForest
from explainer.ShapExplainer import ShapExplainer
from explainer.KernelSHAP import KernelSHAP
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.models import KitNet,HalfSpaceTrees
from evaluation.evaluator import compute_ed1_consistency,explain_sample
from explainer.Lime_method import LIME
import json

if __name__ == "__main__":

    # Load dataset for streaming 
    data = pd.read_csv("datasets/17.csv",delimiter=";", header=0, index_col='datetime', parse_dates=True, squeeze=True)
    
    

    # Pre-processing data
    X_all = data.drop(['anomaly','changepoint'], axis=1)
    y_all = data['anomaly']

    """
        Below, this experiment is for time series not for flow of data
    """
    # test on explainer 
    #record_to_explain = X_all.iloc[14,:]
    #sh = ShapExplainer(num_anomalies_to_explain=8,reconstruction_error_percent=0.7)
    # sh.train_model(X_all,nb_epoch=10)
    # print(sh.get_top_anomaly_to_explain(X_all.head(20)))
    # df_err,total_mse = sh.get_errors_df_per_record(record_to_explain)
    # print(sh.get_num_features_with_highest_reconstruction_error(total_mse*df_err.shape[0],df_err))
    #sh.explain_unsupervised_data(X_all,X_all.head(20),return_shap_values=True)
    """
        End bloc
    """


    # Create a model 
    model = RobustRandomCutForest(num_trees=100, window_size=200, tree_size=256)

    #  Fit a model 
    def fit():
        auroc = AUROCMetric() 
        i = 0
        for X in tqdm(np.array(X_all)):  # Stream data.
            model.fit(X)  # Fit model to and score the instance.
            score = model.score(X)
            # print ("Instance",i," anomaly score",score)
            i += 1
            for y in y_all:
                auroc.update(y, score)

        print("AUROC Metric:",auroc.get())
    
    
    seuil = 0.999
    
    # Model for anomaly detection
    kitnet_model = KitNet(grace_anomaly_detector=500)

    
    
    # WITH  KernelSHAP
    """
    iterator = ArrayStreamer(shuffle=False)  # Init streamer to simulate streaming data.
    important_fts = []
    list_of_all_shap_values = []
    L1 = []
    idx = 0
    index_list = []
    for X in tqdm(iterator.iter(np.array(X_all))):  # Stream data.
        idx += 1
        score1 =  kitnet_model.fit_score_partial(X) # Fit kitnet_model to and score the instance.
        L1.append(score1)
        # Introduce a shap explainer 
        if score1 > seuil :
            index_list.append(idx)
            anormalous_instance_to_explain = X_all.iloc[idx,:]
            print("\n",idx,"****************")
            explainer = KernelSHAP(X_all)
            shap_values = explainer.train_explainer(kitnet_model.score,anormalous_instance_to_explain)
            list_of_all_shap_values.append(shap_values [0])
            # Explanation for each anomaly 
            explainer.save_plot(kitnet_model.score,idx,X_all,anormalous_instance_to_explain)
    
    # df1 = pd.DataFrame(list_of_all_shap_values, index=index_list,columns=list(X_all.columns))
    
    # print(df1)
    """

    # Evaluation
    # consistency_list = []
    # time_list = []
    # for i in index_list:
    #     time = explain_sample(X_all.iloc[i,:],kitnet_model,X_all,4, sample_labels=None)[1]
    #     consistency = compute_ed1_consistency(instance=X_all.iloc[i,:],model=kitnet_model,X_all=X_all, num_features=4,instance_labels=y_all, 
    #                             explanation=explain_sample(X_all.iloc[i,:],kitnet_model,X_all,4, sample_labels=None)[0], normalized=True )
    #     consistency = round(consistency,2)
    #     consistency_list.append(consistency)
    #     time_list.append(time)
    
    # d = {'Index':index_list,'Score':consistency_list,'Time(s)':time_list}
    # evaluation_scores =  pd.DataFrame(d)
    # print(evaluation_scores)
    
    # End Evaluation
    
    
    


    # With LIME
    iterator = ArrayStreamer(shuffle=False)  # Init streamer to simulate streaming data.
    important_fts = []
    L1 = []
    idx = 0
    feature_names = list(X_all.columns)
    index_list = []
    dic_explanations = {}
    for X in tqdm(iterator.iter(np.array(X_all))):  # Stream data.
        idx += 1
        score1 =  kitnet_model.fit_score_partial(X) # Fit kitnet_model to and score the instance.
        L1.append(score1)   
        # Introduce a LIME explainer 
        if score1 > seuil :
            index_list.append(idx)
            anormalous_instance_to_explain = X_all.iloc[idx,:]
            print("\n",idx,"****************")
            explainer_lime = LIME()
            explainer_lime.fit(X_all.head(100).values)
            explainer_lime.save_plot(anormalous_instance_to_explain, kitnet_model.score)
            # Explanation for each anomaly 
            # explanation = explainer_lime.explain_sample(anormalous_instance_to_explain,kitnet_model.score)[0]
            # name_of_important_features = []
            # for c in explanation['important_fts']:
            #     name_of_important_features.append(feature_names[int(c)])
            # dic_explanations[str(anormalous_instance_to_explain.name)] = name_of_important_features

    # Evaluation
    # consistency_list = []
    # time_list = []
    # for i in index_list:
    #     time = explain_sample(X_all.iloc[i,:],kitnet_model,X_all,4, sample_labels=None)[1]
    #     consistency = compute_ed1_consistency(instance=X_all.iloc[i,:],model=kitnet_model,X_all=X_all, num_features=4,instance_labels=y_all, 
    #                             explanation=explain_sample(X_all.iloc[i,:],kitnet_model,X_all,4, sample_labels=None)[0], normalized=True )
    #     consistency = round(consistency,2)
    #     consistency_list.append(consistency)
    #     time_list.append(time)
    
    # d = {'Index':index_list,'Score':consistency_list,'Time(s)':time_list}
    # evaluation_scores =  pd.DataFrame(d)
    # print(evaluation_scores)
    
    #print(dic_ecplanations)
    # with open('explanation_lime.json', 'w') as jsonFile:
    #     json.dump(dic_explanations, jsonFile,indent=4)
    
    # End with LIME
          

    def stability(nb_tests):
        L = dict()
        list_index_of_df = []

        for i in range(nb_tests):
            print("Test ",i)
            iterator = ArrayStreamer(shuffle=False)  # Init streamer to simulate streaming data.
            important_fts = []
            list_of_all_shap_values = []
            L1 = []
            idx = 0
            index_list = []
            for X in tqdm(iterator.iter(np.array(X_all))):  # Stream data.
                idx += 1
                score1 =  kitnet_model.fit_score_partial(X) # Fit kitnet_model to and score the instance.
                L1.append(score1)
                # Introduce a shap explainer 
                if score1 > seuil :
                    index_list.append(idx)
                    anormalous_instance_to_explain = X_all.iloc[idx,:]
                    print("\n",idx,"****************")
                    explainer = KernelSHAP(X_all)
                    shap_values = explainer.train_explainer(kitnet_model.score,anormalous_instance_to_explain)
                    list_of_all_shap_values.append(shap_values [0])
                    # Explanation for each anomaly 
                    explainer.save_plot(kitnet_model.score,idx,X_all,anormalous_instance_to_explain)
            
            name = f"df_{i}"
            list_index_of_df.append(name)

            #print(name)
            L[name] = pd.DataFrame(list_of_all_shap_values, index=index_list,columns=list(X_all.columns))
        
        print("$$$$$$",L)

        stability_explain = 0
        for i in range(nb_tests):
            if L[list_index_of_df[i%nb_tests]].equals(L[list_index_of_df[(i+1)%nb_tests ]]):
                stability_explain = 0 
            else:
                stability_explain += 1

        if stability_explain == 0:
            return "\nStabilité des explications!!!"
        else:
            return "\nInstabilité des explications!!!"    

    # print(stability(3))                  

    def explain_instance_fully(detector,seuil, data, explainer_type="SHAP"):
        iterator = ArrayStreamer(shuffle=False)  # Init streamer to simulate streaming data.
        L1 = []
        idx = 0
        index_list = []
        for X in tqdm(iterator.iter(np.array(data))):  # Stream data.
            idx += 1
            score1 =  detector.fit_score_partial(X) # Fit detector to and score the instance.
            L1.append(score1)
            if score1 > seuil :
                index_list.append(idx)
                anormalous_instance_to_explain = data.iloc[idx,:]
                print("\n",idx,"****************")
                # Introduce a shap explainer 
                if explainer_type == "SHAP":
                    explainer = KernelSHAP(data)
                    explainer.save_plot(detector.score,idx,data,anormalous_instance_to_explain)
                # Introduce a lime explainer 
                if explainer_type == "LIME":
                    explainer_lime = LIME()
                    explainer_lime.fit(X_all.head(100).values)
                    explainer_lime.save_plot(anormalous_instance_to_explain, detector.score)
    

    def explain_instance_specific(detector, data, idx, explainer_type="SHAP"):
        anormalous_instance_to_explain = data.iloc[idx,:]
        # Introduce a shap explainer 
        if explainer_type == "SHAP":
            explainer = KernelSHAP(data)
            explainer.save_plot(detector.score,idx,data,anormalous_instance_to_explain)
        # Introduce a lime explainer 
        if explainer_type == "LIME":
            explainer_lime = LIME()
            explainer_lime.fit(X_all.head(100).values)
            explainer_lime.save_plot(anormalous_instance_to_explain, detector.score)


    
    # explain_instance_fully(kitnet_model,0.999, X_all, explainer_type="SHAP")
    #explain_instance_specific(kitnet_model,X_all,1069, explainer_type="LIME")
    

    def pfeiffer(seuil):
        df = pd.read_csv('datasets/Components_final.csv',index_col='timestamp')
        df = df.drop(['Group'], axis=1)
        model = KitNet(grace_anomaly_detector=1000)
        explain_instance_fully(model,seuil, df, explainer_type="SHAP")


    #pfeiffer(25)

        

