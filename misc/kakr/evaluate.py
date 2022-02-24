from tensoflow.keras import backend as K

def _recall(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_result = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positive / (actual_result + K.epsilon())

def _precision(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_result = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positive / (predicted_result + K.epsilon())

def _f1_score(y_true, y_pred):
    precision = precision_(y_true, y_pred)
    recall = recall_(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
