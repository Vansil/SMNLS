import pandas as pd
import torch


def sent_eval_table(results_file, output_file):
    '''
    Outputs an image displaying SentEval results in a table
    Args:
        results_file
        output_file: html file name
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