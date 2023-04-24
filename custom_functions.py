from sklearn.metrics import * 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import pandas as pd

# Generate confusion matrix
def plot_conf_mat(clf, X_test, y_test):
    matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues)
    plt.title('Confusion matrix for our classifier')
    plt.show(matrix)
    plt.show()
    
def plot_roc_curve(testLabels, predictionProbabilities, pos_label = 1):
    fpr, tpr, thresholds = roc_curve(testLabels, predictionProbabilities, pos_label = 1)
    # calculate scores
    lr_auc = roc_auc_score(testLabels, predictionProbabilities)
    print('AUC Score = %.3f' % (lr_auc * 100))
    plt.rcParams['figure.figsize'] = [7, 7]
    
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title('Feature Importance Plot for ' + model_type)
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    
def calculateMetricsAndPrint(predictions, predictionsProbabilities, actualLabels):
    predictionsProbabilities = [item[1] for item in predictionsProbabilities]
    
    accuracy = accuracy_score(actualLabels, predictions) * 100
    precisionNegative = precision_score(actualLabels, predictions, average = None)[0] * 100
    precisionPositive = precision_score(actualLabels, predictions, average = None)[1] * 100
    recallNegative = recall_score(actualLabels, predictions, average = None)[0] * 100
    recallPositive = recall_score(actualLabels, predictions, average = None)[1] * 100
    auc = roc_auc_score(actualLabels, predictionsProbabilities) * 100
    
    print("Accuracy: %.2f\nPrecisionNegative: %.2f\nPrecisionPositive: %.2f\nRecallNegative: %.2f\nRecallPositive: %.2f\nAUC Score: %.2f\n" % 
          (accuracy, precisionNegative, precisionPositive, recallNegative, recallPositive, auc))