import src.util.ap_cal as ap_calculator
import numpy as np


def top_k_by_class(predictions, labels, k=20):
  """Extracts the top k predictions for each video, sorted by class.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    k: the top k non-zero entries to preserve in each prediction.

  Returns:
    A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
    are lists of lists of floats. 'true_positives' is a list of scalars. The
    length of the lists are equal to the number of classes. The entries in the
    predictions variable are probability predictions, and
    the corresponding entries in the labels variable are the ground truth for
    those predictions. The entries in 'true_positives' are the number of true
    positives for each class in the ground truth.

  Raises:
    ValueError: An error occurred when the k is not a positive integer.
  """
  if k <= 0:
    raise ValueError("k must be a positive integer.")
  k = min(k, predictions.shape[1])
  num_classes = predictions.shape[1]
  prediction_triplets= []
  for video_index in range(predictions.shape[0]):
    prediction_triplets.extend(top_k_triplets(predictions[video_index],labels[video_index], k))
  out_predictions = [[] for v in range(num_classes)]
  out_labels = [[] for v in range(num_classes)]
  for triplet in prediction_triplets:
    out_predictions[triplet[0]].append(triplet[1])
    out_labels[triplet[0]].append(triplet[2])
  if type(labels) is np.ndarray:
      out_true_positives = [np.sum(labels[:,i]) for i in range(num_classes)]
  else:
      out_true_positives = [np.sum(labels.numpy()[:,i]) for i in range(num_classes)]

  return out_predictions, out_labels, out_true_positives


def top_k_triplets(predictions, labels, k=20):
  """Get the top_k for a 1-d numpy array. Returns a sparse list of tuples in
  (prediction, class) format"""
  m = len(predictions)
  k = min(k, m)
  indices = np.argpartition(predictions, -k)[-k:]
  return [(index, predictions[index], labels[index]) for index in indices]

def flatten(l):
  """ Merges a list of lists into a single list. """
  return [item for sublist in l for item in sublist]

def calculate_gap(predictions, actuals, top_k=20):
  """Performs a local (numpy) calculation of the global average precision.

  Only the top_k predictions are taken for each of the videos.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.
    top_k: How many predictions to use per video.

  Returns:
    float: The global average precision.
  """
  gap_calculator = ap_calculator.AveragePrecisionCalculator()
  sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, actuals, top_k)
  gap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
  return gap_calculator.peek_ap_at_n()