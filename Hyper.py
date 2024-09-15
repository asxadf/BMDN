Data_Train_Path = '/Applications/Project_PyCharm/DOE_CHS_Paper/Data_Backup/Dataset_Train_Matlab_Look_Ahead_4_Month_2010_to_2017.csv'
# Data_Test_Path  = '/Applications/Project_PyCharm/DOE_CHS_Paper/Data_Backup/Dataset_Test.csv'
Data_Test_Path  = '/Applications/Project_PyCharm/DOE_CHS_Paper/Data_Backup/Dataset_Test_Matlab_Look_Ahead_4_Month_2018.csv'


Learning_Rate = 0.001
Hidden_Unit_Structure = [4, 4]
Activation_Type = 'sigmoid'
Date_Format = '%Y/%m/%d'
Feature_Score_Threshold = 0.6

Num_Label = 16 # Look-ahaed 4 monthes or 2 monthes
Num_Batch = 10
Num_Branch = 2
Num_Component = 2
Num_Sampling = 20 #20
Num_Epoch = 300 #300
Num_Removed_IMF = 1

Name_Feature = ['Temperature_C',
                'Temperature_D',
                'WI_Speed_D',
                'WI_Volumn_D',
                'Temperature_M',
                'WI_Volumn_of_Dam',
                'SST_Nino12',
                'SST_Nino3',
                'SST_Nino34']

