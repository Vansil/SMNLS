import pandas as pd
import torch

'''
Ideas:
+ WiC ROC curve
+ t-sne of sentence batch under model
'''

def sent_eval_table(results_file, output_file):
    '''
    Outputs an image displaying SentEval results in a table
    Args:
        results_file
        output_file: html file name
    Returns:
        pandas table
    '''
    # Load results
    results = torch.load(results_file)['senteval']
    
    # Extract relevant scores
    task_names = []
    metric_names = []
    scores = []
    for name in results:
        task_names.append(name)
        metric_names.append('acc')
        scores.append(results[name][0]) # probably task dependent
    
    # Make table
    table = pd.DataFrame({
        'Task': task_names,
        'Metric': metric_names,
        'Score': scores
    })

    # Output to file
    table.to_html(output_file)
    return table


def wic_table(results_files_dict, output_file, include_thresholds=False, include_train_acc=False):
    '''
    Makes html table comparing WiC accuracies.
    Args:
        results_files_dict: dictionary with keys model name, values result file path of the model
        output_file: html table is written here
        include_thresholds: set to True to include best performing threshold
        include_train_acc: set to True to include best training accuracy
    Returns:
        pandas table
    '''
    # Load data from result files
    names = []
    test_accs = []
    train_accs = []
    thresholds = []
    for name, path in results_files_dict.items():
        results = torch.load(path)['wic']
        names.append(name)
        test_accs.append("{:.1f}%".format(results['test_accuracy']*100))
        train_accs.append("{:.1f}%".format(results['train_accuracy']*100))
        thresholds.append("{:.2f}%".format(results['threshold']))
    
    # Make table
    frame = {'Model': names}
    if include_thresholds:
        frame['Threshold'] = thresholds
    if include_train_acc:
        frame['Train acc'] = train_accs
    frame['Test acc'] = test_accs

    # Output to file
    table = pd.DataFrame(frame)
    table.to_html(output_file)
    return table