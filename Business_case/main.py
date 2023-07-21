import numpy as np
from sklearn import preprocessing # To standardize the inputs scikit-learn

cvs_data_url = "https://raw.githubusercontent.com/ingcastaleo/DataScience365_Udemy/main/Business_case/Audiobooks_data.csv"
cvs_data_url = "Business_case\src\Audiobooks_data.csv"
raw_csv_data = np.loadtxt(cvs_data_url,delimiter=',')