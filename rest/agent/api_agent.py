import requests


def post(dictionary, modality_type, remote_ip):
    """
    Sends the dataset to  :param remote_ip
    :param dictionary: data to be sent
    :param modality_type: one of ['eeg', 'emg' or 'ecg']
    :param remote_ip: IP of the remote server (String)
    :return: a response {'SUCCESS': <SUCCESS_CODE>}
    """
    url = 'http://{}:5000/{}/data'
    return requests.post(url.format(str(remote_ip), str(modality_type)),
                         json={str(modality_type): dictionary},
                         headers={'content-type': 'application/json', 'Accept': 'application/json'})
