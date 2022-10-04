import shap
import time
import numpy as np
from explainer.KernelSHAP import KernelSHAP
import random
from scipy.stats import entropy
from math import floor, ceil
from explainer.Lime_method import LIME

array_feature_names = np.array(['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature','Thermocouple', 'Voltage','Volume Flow RateRMS'])

"""
  L'explication doit être localement fidèle. Dans notre contexte, 
  cela signifie que la même explication peut s'appliquer à des "voisins" immédiats, 
  qui sont des instances d'anomalies de la même application et du même type anormal, 
  et autour de la même période.

  Nous procédons comme suit:
    *Nous créerons une procédure de sous échantillonnage autour de l’instance à expliquer qui génère un ensemble d'échantillons que nous appelons { X }
    *Nous définissons les explications correspondantes pour chaque échantillon dans un ensemble { F }
    *Nous appliquons une fonction d’extraction sur cet ensemble d’explication, celle-ci permet d’extraire les caractéristiques importantes
    *Nous mesurons l’entropie sur la sortie de la fonction d’extraction
"""
def explain_sample_with_lime(sample,model,X_all,num_features,sample_labels=None):
    """Returns the explanation of a sample when we use LIME explainer"""
    explainer = LIME()
    explainer.fit(X_all.head(100).values)
    # Explanation for each anomaly 
    explanation = explainer.explain_sample(sample,model.score)
    return explanation

def explain_sample(sample,model,X_all,num_features,sample_labels=None):
    
    #Returns the explanation of a sample

    #model :  the anomaly detection model
    #num_features :  number of features we are going to consider
    

    start_t = time.time()
    explainer = KernelSHAP(X_all)
    explanation = explainer.train_explainer(model.score,sample)[0]

    # important feature indices are extracted from the feature names reported in the explanation
    #important_fts = []
    feature_ordered = np.argsort(-np.abs(explanation))
    features_important_for_instance= array_feature_names[feature_ordered]
    #important_fts.append(features_important_for_instance)

    # Or can just return only the indexes
    # print("The important features for this innstance are;",features_important_for_instance[:4])

    return {'important_fts' : features_important_for_instance[:num_features]}, time.time() - start_t 

def explain_all_samples(samples,model,X_all,num_features):
    """Returns the explanation and explanation time for all the provided `samples` together.

    We define the shared explanation of a set of samples as the (duplicate-free) union
    of the "important" explanatory features found for each sample.

    Indeed, for a single sample, these important features constitute those that affected
    the outlier score function the most locally. Taking the union of these features hence
    gives us a sense of the features that were found most relevant throughout the samples.

    Args:
        samples (ndarray): samples to explain of shape `(n_samples, sample_length, n_features)`.

    Returns:
        dict, float: explanation dictionary and explanation time (in seconds) for the samples.
    """
    samples_explanation, samples_time = {'important_fts': set()}, 0
    for sample in samples:
        explanation, time = explain_sample(sample, model, X_all, num_features)
        # print("$$$$$$$$$$$$",samples_explanation)
        samples_explanation['important_fts'] = samples_explanation['important_fts'].union(
            explanation['important_fts']
        )
        samples_time += time
    return samples_explanation, samples_time

def explain_each_sample(samples,model,X_all, num_features, samples_labels=None):
    """Returns the explanation and explanation time for each of the provided `samples`.

    Args:
        samples (ndarray): samples to explain of shape `(n_samples, sample_length, n_features)`.
            With `sample_length` possibly depending on the sample.
        samples_labels (ndarray): optional samples labels of shape `(n_samples, sample_length)`.
            With `sample_length` possibly depending on the sample.

    Returns:
        list, list: list of explanation dictionaries and corresponding times.
    """
    explanations, times = [], []
    if samples_labels is None:
        samples_labels = np.full(len(samples), None)
    for sample, sample_labels in zip(samples, samples_labels):
        explanation, time = explain_sample(sample,model,X_all, num_features, sample_labels)
        explanations.append(explanation)
        times.append(time)
    return explanations, times

def get_sliding_windows(array, window_size, window_step, include_remainder=False, dtype=np.float64):
    """Returns sliding windows of `window_size` elements extracted every `window_step` from `array`.

    Args:
        array (ndarray): input ndarray whose first axis will be used for extracting windows.
        window_size (int): number of elements of the windows to extract.
        window_step (int): step size between two adjacent windows to extract.
        include_remainder (bool): whether to include any remaining window, with a different step size.
        dtype (type): optional data type to enforce for the window elements (default np.float64).

    Returns:
        ndarray: windows array of shape `(n_windows, window_size, *array.shape[1:])`.
    """
    windows, start_idx = [], 0
    while start_idx <= array.shape[0] - window_size:
        # print(start_idx,"****",array.shape[0] - window_size)
        windows.append(array[start_idx:(start_idx + window_size)])
        start_idx += window_step
    if include_remainder and start_idx - window_step + window_size != array.shape[0]:   
        windows.append(array[-window_size:])
    return np.array(windows, dtype=dtype)

ed1_consistency_n_disturbances = 5
min_instance_length = 3
large_anomalies_coverage = 'all'

def get_disturbances_features(instance, model,X_all, num_features, instance_labels, explanation=None):
  """Model-dependent implementation.

    Returns the explanatory features found for different "disturbances" of `instance`.

    The number of disturbances to perform is given by the value of

    `ed1_consistency_n_disturbances`, which might include the original instance or not.



  Args:
      instance (ndarray): instance data of shape `(instance_length, )`.
      instance_labels (ndarray): instance labels of shape `(instance_length,)`.
      explanation (dict|None): optional, pre-computed, explanation of `instance`.


  For model-dependent explainers, "disturbances" are defined differently depending on
  the instance length and anomaly coverage policy:

  - For instances of minimal length, disturbances are defined as 1-step sliding windows to the right.
  - For others, disturbances are defined depending on the anomaly coverage policy:
      - "Center" coverage: as 1-step sliding windows* alternately to the right and left.
      - "End" coverage: as 1-step sliding windows to the left.
      - "All" coverage: as random samples of `n` windows among all possible 1-step sliding ones
          in the instance. Where `n` is the number of windows used when explaining the original instance.

  *"windows" are all of size `explainer.sample_length`.

  In all above cases, the explanatory features of the original instance are (by convention)
  included in the consistency computation.



  Returns:
      list: the list of explanatory features lists for each instance "disturbance".

  """
  """normality_model (Forecaster|Reconstructor): "normality model" whose predictions will be
            #used to derive outlier scores (for now either a `Forecaster` or `Reconstructor` object.
  # explainer.sample_length  can be window size
  sample_length dans ModelDependentExplainer(Explainer)
  """
  a = 3 # à revoir

  instance_length, sample_length = len(instance), a
  n_disturbances = ed1_consistency_n_disturbances
  if instance_length == min_instance_length:
      # instance of minimal length: no initialization and slide through the whole instance
      samples_fts = []
      sub_instance = instance
  else:
      # larger instance: initialize with the explanation of the original instance
      e = explanation if explanation is not None else explain_sample(instance,model,X_all, num_features, instance_labels)[0]
       # print("*******************",e)
      samples_fts = [e['important_fts']]

      # print("First step output ", samples_fts)
      if large_anomalies_coverage == 'end':
          # slide through the end of the instance if "end" anomaly coverage
          sub_instance = instance[(instance_length - sample_length - n_disturbances + 1):-1]
  

  if instance_length == min_instance_length or large_anomalies_coverage == 'end':
      samples = get_sliding_windows(sub_instance, sample_length, 1)
      samples_explanations, _ = explain_each_sample(samples,model,X_all,num_features)
      samples_fts += [s_e['important_fts'] for s_e in samples_explanations]
  elif large_anomalies_coverage == 'center':
      # alternately slide to the right and left of the anomaly center if "center" anomaly coverage
      center_start = floor((instance_length - sample_length) / 2)
      slide, slide_sign = 1, 1
      for _ in range(ed1_consistency_n_disturbances - 1):
          start = center_start + slide_sign * slide
          sample_explanation, _ = explain_sample(instance[start:(start + sample_length)],model,X_all , num_features)
          samples_fts.append(sample_explanation['important_fts'])
          slide_sign *= -1
          if slide_sign == 1:
              slide += 1
  else:
      # construct every possible samples from the instance if "all" anomaly coverage

      # petit test
      samples_pool = get_sliding_windows(instance, sample_length, 1)
      array_zeros = np.zeros((samples_pool.shape[0],len(instance) -samples_pool.shape[1]))
      tab_rempli = np.concatenate((samples_pool, array_zeros), axis=1) 

      ###
      # randomly sample the same number of samples as used for explaining the original instance
      for _ in range(n_disturbances):
          # any remaining sample was included when explaining the original instance
          samples = tab_rempli[
              random.sample(range(len(samples_pool)), ceil(instance_length / sample_length))
          ]
          samples_explanation, _ = explain_all_samples(samples, model,X_all, num_features)
          samples_fts.append(samples_explanation['important_fts'])
      # print(samples_fts)
  return samples_fts


def compute_features_consistency(features_lists):
    """Returns the "consistency" of the provided features lists.

    This "consistency" aims to capture the degree of agreement between the features lists.
    We define it as the entropy of the lists' duplicate-preserving union (i.e., turning the
    features lists into a features bag).

    The smaller the entropy value, the less uncertain will be the outcome of randomly
    drawing an explanatory feature from the bag, and hence the more the lists of features
    will agree with each other.

    Args:
        features_lists (list): list of features lists whose consistency to compute.

    Returns:
        float: consistency of the features lists.
    """
    features_bag = [ft for features_list in features_lists for ft in features_list]
    # unnormalized probability distribution of feature ids
    p_features = []
    for feature_id in set(features_bag):
        p_features.append(features_bag.count(feature_id))
    return entropy(p_features, base=2)

def compute_ed1_consistency(instance, model,X_all, num_features, instance_labels, explanation=None, normalized=True):
    """Returns the ED1 consistency (i.e., stability) score of the provided `instance`.
    This metric is defined as the (possibly normalized) consistency of explanatory
    features across different "disturbances" of the explained instance.
    Args:
        instance (ndarray): instance data of shape `(instance_length, n_features)`.
        instance_labels (ndarray): instance labels of shape `(instance_length,)`.
        explanation (dict|None): optional instance explanation, that can be used to normalize
            the consistency score of the instance.
        normalized (bool): whether to "normalize" the consistency score with respect to the
            original explanation's conciseness (if True, `explanation` must be provided).
    Returns:
        float: the ED1 consistency score of the instance.
    """
    if normalized:
        a_t = 'normalizing consistency requires providing the instance explanation'
        assert explanation is not None, a_t
        #print(a_t)
    # consistency of explanatory features across different instance "disturbances"
    fts_consistency = compute_features_consistency(
        get_disturbances_features(instance,model,X_all, num_features, instance_labels, explanation)
    )
    if not normalized:
        return fts_consistency

    return (2 ** fts_consistency) / len(explanation['important_fts'])

"""
    For stability, we need to run our explanation model many times.
    We are going to see if all explanations are similar during each test.
"""