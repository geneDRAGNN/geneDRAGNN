
import numpy as np
import json

from sklearn.metrics import classification_report
def evaluate_model(model, data, logger=None):
    '''returns train and test classification reports'''

    model.to(device='cuda')
    logits, _, _ = model.forward(data.to(device='cuda'))

    preds_train = logits[data.train_mask].argmax(dim=-1).cpu().detach().numpy()
    preds_test = logits[data.test_mask].argmax(dim=-1).cpu().detach().numpy()

    y_train = data.y[data.train_mask].cpu().detach().numpy()
    y_test = data.y[data.test_mask].cpu().detach().numpy()

    train_report = classification_report(y_train, preds_train, labels=[0,1], target_names=['negative', 'positive'],
                                         output_dict=True)
    test_report = classification_report(y_test, preds_test, labels=[0,1], target_names=['negative', 'positive'],
                                        output_dict=True)
    if logger:
        train_acc = train_report['accuracy']
        test_acc = test_report['accuracy']
        logger.log_metrics({'final_train_acc': train_acc, 'final_test_acc': test_acc})
        logger.log_metrics({'train_report': train_report, 'test_report': test_report})

    return train_report, test_report

def save_reports(filename, train_reports, test_reports):
    '''saves train and test reports to a json file'''
    save_dict = {'train_reports': train_reports, "test_reports": test_reports}
    json_string = json.dumps(save_dict)
    json_file = open(f'{filename}.json', 'w')
    json_file.write(json_string)
    json_file.close()


def get_value_from_dict(report, *keys):
    '''gets value from dict through sequence of keys'''
    value = report
    for key in keys:
        value = value[key]
    return value

def average_report_val(reports, *keys):
    '''averages a particular report value over a list of reports'''
    return np.average([get_value_from_dict(report, *keys) for report in reports])
