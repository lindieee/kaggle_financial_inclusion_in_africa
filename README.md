# Financial inclusion in Africa


## Project description

This project is based on a Kaggle competition about predicting if people living in Africa have a bank account or not, based on factors like education level, marital status, household size etc.

[Kaggle: Financial Inclusion in Africa](https://zindi.africa/competitions/financial-inclusion-in-africa)


### Model: 
- train binary classifier: bank account Yes/No
- best model: 
    - Max-Voting-Classifier
    - base models: logistic regression, xgboost, random forest
    - best hyperparameters for random forest were found via grid search + cross validation
    - class_weight=balanced (due to high class imbalance)
    - soft voting (works much better than hard voting)

### Feature engineering:
- feature selection (take only some of the given indicators)
- one hot encoding of categorical features

<br>


## Set up Environment


### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```


## Get data

Get data from [here](https://zindi.africa/competitions/financial-inclusion-in-africa/data) and store it in data folder.



## Usage

Train model, store test data in data folder and model in models folder:

```bash
python train.py  
```

Test predict.py on test set:

```bash
python predict.py models/max_voting_classifier.sav data/X_test.csv data/y_test.csv
```

## Model Performance

### Train data


                   precision  recall   f1-score   support

           0       0.94        0.92     0.93       15154
           1       0.57        0.66     0.61       2483
           

    Mean Absolute Error = 0.1188



### Test data

                   precision  recall   f1-score   support

           0       0.94        0.91     0.92       5052
           1       0.53        0.62     0.57       827
            
    Mean Absolute Error = 0.1323



