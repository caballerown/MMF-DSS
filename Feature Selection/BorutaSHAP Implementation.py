# Caballero, Gaw, Jenkins, and Johnstone
# Toward Automated Instructor Pilots in Legacy Air Force Systems: 
# Physiology-based Flight Difficulty Classification via Machine Learning
# BorutaSHAP 
# Physiological features only

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from BorutaShap import BorutaShap
#matplotlib inline

# Import input features 
input_df = pd.read_csv('total_dev_all_phys.csv') 

# Drop unnessary columns
input_df = input_df.drop(['Subject-Run'], axis=1) 

# Import response/output values
output_df = pd.read_csv('PerfMetrics_dev.csv')
difficulty = output_df['Difficulty']

# Feature selection via BorutaShap
Feature_Selector = BorutaShap(importance_measure='shap',classification = True)
Feature_Selector.fit(X=input_df, y = difficulty, n_trials = 100, random_state=42)
input_df = Feature_Selector.Subset()

# Assign training data 
x_train = input_df
y_train = difficulty

# load test data
x_test = pd.read_csv('total_val_all_phys.csv') 
x_test = x_test.loc[:, input_df.columns] 

output_df_test = pd.read_csv('PerfMetrics_val.csv')
difficulty_test = output_df_test['Difficulty']
y_test = difficulty_test

# Convert to binary
y_train_binary = y_train.copy()
y_train_binary[:] = [x if x != 2 else 1 for x in y_train_binary]
y_train_binary[:] = [x if x != 3 else 4 for x in y_train_binary]

y_test_binary = y_test.copy()
y_test_binary[:] = [x if x != 2 else 1 for x in y_test_binary]
y_test_binary[:] = [x if x != 3 else 4 for x in y_test_binary]

# Save files
x_train.to_csv(r'C:\Users\masked_user\Desktop\xtrain.csv', index=False, header=True)
y_train.to_csv(r'C:\Users\masked_user\Desktop\ytrain.csv', index=False, header=True)
x_test.to_csv(r'C:\Users\masked_user\Desktop\xtest.csv', index=False, header=True)
y_test.to_csv(r'C:\Users\masked_user\Desktop\ytest.csv', index=False, header=True)



# Caballero, Gaw, Jenkins, and Johnstone
# Toward Automated Instructor Pilots in Legacy Air Force Systems: 
# Physiology-based Flight Difficulty Classification via Machine Learning
# BorutaSHAP 
# All features included 

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from BorutaShap import BorutaShap
#matplotlib inline

# Import input features 
input_df = pd.read_csv('total_dev_all.csv') 

# Drop unnessary columns
input_df = input_df.drop(['Subject-Run'], axis=1) 

# Import response/output values
output_df = pd.read_csv('PerfMetrics_dev.csv')
difficulty = output_df['Difficulty']

# Feature selection via BorutaShap
Feature_Selector = BorutaShap(importance_measure='shap',classification = True)
Feature_Selector.fit(X=input_df, y = difficulty, n_trials = 100, random_state=42)
input_df = Feature_Selector.Subset()


# Assign training data 
x_train = input_df
y_train = difficulty

# load test data
x_test = pd.read_csv('total_val_all.csv') 
x_test = x_test.loc[:, input_df.columns] 

output_df_test = pd.read_csv('PerfMetrics_val.csv')
difficulty_test = output_df_test['Difficulty']
y_test = difficulty_test

# Convert to binary
y_train_binary = y_train.copy()
y_train_binary[:] = [x if x != 2 else 1 for x in y_train_binary]
y_train_binary[:] = [x if x != 3 else 4 for x in y_train_binary]

y_test_binary = y_test.copy()
y_test_binary[:] = [x if x != 2 else 1 for x in y_test_binary]
y_test_binary[:] = [x if x != 3 else 4 for x in y_test_binary]

# Save files
x_train.to_csv(r'C:\Users\masked_user\Desktop\xtrain2.csv', index=False, header=True)
y_train.to_csv(r'C:\Users\masked_user\Desktop\ytrain2.csv', index=False, header=True)
x_test.to_csv(r'C:\Users\masked_user\Desktop\xtest2.csv', index=False, header=True)
y_test.to_csv(r'C:\Users\masked_user\Desktop\ytest2.csv', index=False, header=True)

