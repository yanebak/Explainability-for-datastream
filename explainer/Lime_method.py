
from lime import lime_tabular
import time
import re
import warnings
import json

feature_names1 = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature','Thermocouple', 'Voltage','Volume Flow RateRMS']


class LIME:
    """Local Interpretable Model-agnostic Explanations (LIME) explanation discovery class.

    For this explainer, the `fit` method must be called before explaining samples.

    See https://arxiv.org/pdf/1602.04938.pdf for more details.
    """
    def __init__(self,output_path=None,n_features=5, ad_model=None):
        # number of features to report in the explanations
        self.output_path = output_path
        self.n_features = n_features
        # LIME model
        self.lime_model = None
    
    def fit (self,training_samples):
        """Initializes and fits the LIME model to the provided `training_samples`.

        We use the implementation of the LIME package with continuous features discretized into
        deciles. `RecurrentTabularExplainer` is used to accommodate to the samples shape.

        Args:
            training_samples (ndarray): training samples of shape `(n_samples, sample_length, n_features)`.
        """
        # feature names must be consistent with the way we identify features in `explain_sample`
        #feature_names = list(training_samples.columns)
        feature_names = [f'ft_{i}' for i in range(training_samples.shape[1])]

        self.lime_model = lime_tabular.LimeTabularExplainer(
            training_samples, mode='regression', feature_names=feature_names,
            discretize_continuous=True, discretizer='decile'
        )
        #print("Successfuly fiited!!")

    def explain_sample(self, sample,function_score, sample_labels=None):
        """LIME implementation.

        An "explanation" is defined as a set of relevant features, it is derived from
        the LIME model. Sample labels are not relevant in this case.
        """
        start_t = time.time()
        explanation = self.lime_model.explain_instance(
                sample, function_score, num_features=self.n_features
            )
        try:
            explanation = self.lime_model.explain_instance(
                sample, function_score, num_features=self.n_features
            )
        except AttributeError:
            print('No model initialization, please call the `fit` method before using this explainer')
            return {'important_fts': []}, 0
        # important feature indices are extracted from the feature names reported in the explanation
        important_fts = []
        for statement, weight in explanation.as_list():
            #print(statement)
            # "ft_{id}" names were used to identify numbers that relate to features in the strings
            ft_name = re.findall(r'ft_\d+', statement)[0]
            #print(ft_name)
            ft_index = int(ft_name[3:])
            important_fts.append(ft_index)
        # a same feature can occur in multiple statements
        returned_fts = sorted(list(set(important_fts)))
        if len(returned_fts) == 0:
            warnings.warn('No explanation found for the sample.')
        return {'important_fts': returned_fts}, time.time() - start_t


    def save_plot(self, sample, function_score,dic_explanations={}):
        explanation = self.explain_sample(sample, function_score)[0]
        name_of_important_features = []
        for c in explanation['important_fts']:
            name_of_important_features.append(feature_names1[int(c)])
        dic_explanations[str(sample.name)] = name_of_important_features
        with open('explanation_lime.json', 'w') as jsonFile:
            json.dump(dic_explanations, jsonFile,indent=6)







