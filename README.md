# Multimodal Drowsiness Detection using EEG, EMG, ECG

This repository contains the dataset, procedure and results of our model.

<div align="center">
    <img src="docs/architecture.png" />
</div> 

## Setting Up

WRITE THE SETTING UP SECTION HERE.

## API Endpoints

``curl -d '{"data":"[100, 200, 300]"}' -H "Content-Type: application/json" -X POST http://192.168.8.101:5000/eeg/data``

### Data Endpoints

```python
import requests

# Data as an array
data = [10, 20, 30, 40]

url = 'http://192.168.8.100:5000/<MODALITY_TYPE>/data'

x = requests.post(url, json = {"eeg": data}, headers={'content-type': 'application/json', 'Accept': 'application/json'})
```

For an example, consider the request below.

```python
import requests

data = [10, 20, 30, 40]

url = 'http://192.168.8.100:5000/eeg/data'

x = requests.post(url, json = {"eeg": data}, headers={'content-type': 'application/json', 'Accept': 'application/json'})
```

### Using the Agent Library

Using the `Agent`, third party applications (`EEG, EMG, ECG` producers) can send the accumulated data to the fusion engine. 

1. Adhere to the proper structure of modules. That is, in the root directory, have an empty file named `__init__.py`. 
2. Place `api_agent.py` inside a folder named `agent`. Create an empty `__init__.py` inside the `agent` folder to adhere to proper module structure of Python.
3. Import the agent by using `from agent.api_agent import *`.
4. Execute the imported `post()` method. For an example

```python

"""
Your code here
"""

response = post([10, 20, 30, 40], "eeg", "192.168.8.100")   # This will send your data to the fusion module

"""
Rest of the code here
"""

```

## References 

WRITE THE REFERENCE SECTION HERE.