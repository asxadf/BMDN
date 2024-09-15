from Main.Autos import Library, Hyper
import matplotlib.pyplot as plt

dic = globals()

# Parameters
Learning_Rate = Hyper.Learning_Rate
Hidden_Unit_Structure = Hyper.Hidden_Unit_Structure
Activation_Type = Hyper.Activation_Type
Num_Sampling = Hyper.Num_Sampling
Name_Feature = Hyper.Name_Feature

# Get data
Num_Label, \
Data_Train_FeaLab_Raw, Data_Test_FeaLab_Raw, \
Num_Data_Train, Num_Data_Test, \
Name_FeaLab_Full, Name_Feature_Full, Name_Label, Datetime_Train_List, Datetime_Test_List, \
Col_Min, Col_Max, \
Data_Train_FeaLab_Full, Data_Train_Feature_Full, Data_Train_Label, \
Data_Test_FeaLab_Full, Data_Test_Feature_Full, Data_Test_Label = Library.Get_Data()

# Get feature
# Option 1
#Data_Train_Feature = Data_Train_Feature_Full[Name_Feature]
#Data_Test_Feature = Data_Test_Feature_Full[Name_Feature]
# Option 2
Name_Feature, Socre_Table, \
Data_Train_Feature, Data_Test_Feature = Library.Feature_Selection(Data_Train_Feature_Full,
                                                                  Data_Train_Label,
                                                                  Data_Test_Feature_Full)

# Do EMD
Data_Train_Feature_Smooth, Data_Train_Label_Smooth = Library.EMD(Data_Train_Feature, Data_Train_Label)

# Train BNN or DNN
BMDN, Train_History_BMDN = Library.Train_BMDN(Data_Train_Feature_Smooth, Data_Train_Label_Smooth)
# DNN, Train_History_DNN = Library.Train_DNN(Data_Train_Feature_Smooth, Data_Train_Label_Smooth)

#
# Test for BMDN or DNN
Eva_Realization,\
Eva_Prediction_Mean_BMDN, Eva_Prediction_Stdv_BMDN, Eva_Prediction_UB_BMDN, Eva_Prediction_LB_BMDN, \
Error_BMDN, A1_MAE_BMDN, A2_MAPE_BMDN, A3_RMSE_BMDN, A4_NSE_BMDN = Library.Predict_and_Evaluate_BMDN(BMDN, Data_Test_Feature, Data_Test_Label, Datetime_Test_List, Col_Max, Col_Min)

# Eva_Realization,\
# Eva_Prediction_Mean_DNN, \
# Error_DNN, MAE_DNN, MAPE_DNN, RMSE_DNN, NSE_DNN = Library.Predict_and_Evaluate_DNN(DNN, Data_Test_Feature, Data_Test_Label, Datetime_Test_List, Col_Max, Col_Min)