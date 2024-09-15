import tensorflow as tf
import pandas as pd
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import emd
import math
from Main.MEMD_all import memd
from Main.Autos import Hyper
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import r_regression

dic = globals()

def Get_Data():
    print('Getting data...')
    # Parameters
    Data_Train_Path = Hyper.Data_Train_Path
    Data_Test_Path = Hyper.Data_Test_Path
    Num_Label = Hyper.Num_Label
    # TraTes/Train/Test ----> FeaLab/Feature/Label
    # Loading
    Data_Train_FeaLab_Raw = pd.read_csv(Data_Train_Path)
    Data_Test_FeaLab_Raw = pd.read_csv(Data_Test_Path)

    Num_Data_Train = len(Data_Train_FeaLab_Raw)
    Num_Data_Test = len(Data_Test_FeaLab_Raw)

    # Get data
    Name_FeaLab_Full = Data_Train_FeaLab_Raw.columns.tolist()
    Name_Feature_Full = Name_FeaLab_Full[5:-Num_Label]
    Name_Label = Name_FeaLab_Full[-Num_Label:]
    Datetime_Train_List = Data_Train_FeaLab_Raw.iloc[:, 0:5]
    Datetime_Test_List = Data_Test_FeaLab_Raw.iloc[:, 0:5]

    # Normalize
    Col_Min = Data_Train_FeaLab_Raw[Name_Feature_Full + Name_Label].min()
    Col_Max = Data_Train_FeaLab_Raw[Name_Feature_Full + Name_Label].max()

    # For training data
    Data_Train_FeaLab_Full = (Data_Train_FeaLab_Raw[Name_Feature_Full + Name_Label] - Col_Min) / (Col_Max - Col_Min)
    Data_Train_Feature_Full = Data_Train_FeaLab_Full[Name_Feature_Full]
    Data_Train_Label = Data_Train_FeaLab_Full[Name_Label]

    # For testing data
    Data_Test_FeaLab_Full = (Data_Test_FeaLab_Raw[Name_Feature_Full + Name_Label] - Col_Min) / (Col_Max - Col_Min)
    Data_Test_Feature_Full = Data_Test_FeaLab_Full[Name_Feature_Full]
    Data_Test_Label = Data_Test_FeaLab_Full[Name_Label]

    # Normalize---v2
    # Col_Max = Data_Train_FeaLab_Raw[Name_Feature_Full + Name_Label].max()
    # Col_Min = Data_Train_FeaLab_Raw[Name_Feature_Full + Name_Label].min()
    #
    # For training data
    # Data_Train_FeaLab_Full = Data_Train_FeaLab_Raw[Name_Feature_Full + Name_Label] /Col_Max
    # Data_Train_Feature_Full = Data_Train_FeaLab_Full[Name_Feature_Full]
    # Data_Train_Label = Data_Train_FeaLab_Full[Name_Label]

    # For testing data
    # Data_Test_FeaLab_Full = Data_Test_FeaLab_Raw[Name_Feature_Full + Name_Label] /Col_Max
    # Data_Test_Feature_Full = Data_Test_FeaLab_Full[Name_Feature_Full]
    # Data_Test_Label = Data_Test_FeaLab_Full[Name_Label]
    # Normalize---v2

    return Num_Label, \
           Data_Train_FeaLab_Raw, Data_Test_FeaLab_Raw, \
           Num_Data_Train, Num_Data_Test, \
           Name_FeaLab_Full, Name_Feature_Full, Name_Label, Datetime_Train_List, Datetime_Test_List, \
           Col_Min, Col_Max, \
           Data_Train_FeaLab_Full, Data_Train_Feature_Full, Data_Train_Label, \
           Data_Test_FeaLab_Full, Data_Test_Feature_Full, Data_Test_Label

def EMD(Data_Train_Feature, Data_Train_Label):
    print('Doing EMD...')
    # Parameters
    Num_Removed_IMF = Hyper.Num_Removed_IMF
    Num_Data_Train = len(Data_Train_Feature)
    Name_Feature = Data_Train_Feature.columns.tolist()
    Name_Label = Data_Train_Label.columns.tolist()
    Data_Train_Feature_Numpy = Data_Train_Feature.to_numpy()
    Data_Train_Label_Numpy = Data_Train_Label.to_numpy()

    Data_Train_Feature_Smooth = pd.DataFrame(data=np.zeros((Num_Data_Train, len(Name_Feature))),
                                             columns=Name_Feature)
    Data_Train_Label_Smooth = pd.DataFrame(data=np.zeros((Num_Data_Train, len(Name_Label))),
                                           columns=Name_Label)

    # EMD for features
    print('Smoothing features...')
    IMF_Feature = memd(Data_Train_Feature_Numpy)
    # Draw IMFs
    for idx in range(len(Name_Feature)):
        dic['IMF_' + Name_Feature[idx]] = IMF_Feature[:, idx, :].transpose()
        emd.plotting.plot_imfs(dic['IMF_' + Name_Feature[idx]], xlabel=Name_Feature[idx])
        plt.show()

    # Draw Before_EMD_ and After_EMD
    fig, ax = plt.subplots(len(Name_Feature), figsize=(40, 40), sharex=True)
    for idx in range(len(Name_Feature)):
        dic['Before_EMD_' + Name_Feature[idx]] = np.sum(dic['IMF_' + Name_Feature[idx]][:, 0:],               axis=1)
        dic['After_EMD_'  + Name_Feature[idx]] = np.sum(dic['IMF_' + Name_Feature[idx]][:, Num_Removed_IMF:], axis=1)
        ax[idx].plot(dic['Before_EMD_' + Name_Feature[idx]], label='Before_EMD')
        ax[idx].plot(dic['After_EMD_'  + Name_Feature[idx]], label='After_EMD')
        ax[idx].set_title(Name_Feature[idx])
        ax[idx].set_ylabel('p. u.')
        ax[idx].legend(loc='upper right')
        plt.ylim([0, 1])
        # Replace
        Data_Train_Feature_Smooth.loc[:, Name_Feature[idx]] = dic['After_EMD_' + Name_Feature[idx]]
    plt.show()

    # EMD for labels
    print('Smoothing labels...')
    IMF_Label = memd(Data_Train_Label_Numpy)
    # Draw IMFs
    for idx in range(len(Name_Label)):
        dic['IMF_' + Name_Label[idx]] = IMF_Label[:, idx, :].transpose()
        emd.plotting.plot_imfs(dic['IMF_' + Name_Label[idx]], xlabel=Name_Label[idx])
        plt.show()
    # Draw Before_EMD_ and After_EMD
    fig, ax = plt.subplots(len(Name_Label), figsize=(40, 40), sharex=True)
    for idx in range(len(Name_Label)):
        dic['Before_EMD_' + Name_Label[idx]] = np.sum(dic['IMF_' + Name_Label[idx]][:, 0:],               axis=1)
        dic['After_EMD_'  + Name_Label[idx]] = np.sum(dic['IMF_' + Name_Label[idx]][:, Num_Removed_IMF:], axis=1)
        ax[idx].plot(dic['Before_EMD_' + Name_Label[idx]], label='Before_EMD')
        ax[idx].plot(dic['After_EMD_'  + Name_Label[idx]], label='After_EMD')
        ax[idx].set_title(Name_Label[idx])
        ax[idx].set_ylabel('p. u.')
        ax[idx].legend(loc='upper right')
        plt.ylim([0, 1])
        # Replace
        Data_Train_Label_Smooth.loc[:, Name_Label[idx]] = dic['After_EMD_' + Name_Label[idx]]
    plt.show()

    return Data_Train_Feature_Smooth, Data_Train_Label_Smooth

def Feature_Selection(Data_Train_Feature_Full, Data_Train_Label, Data_Test_Feature_Full):
    print('Doing feature selecting...')
    # Parameters
    Name_Feature_Full = Data_Train_Feature_Full.columns.tolist()
    Name_Label = Data_Train_Label.columns.tolist()

    Feature_Score_Threshold = Hyper.Feature_Score_Threshold

    # Step 1: Score
    Socre_Table = pd.DataFrame(data=np.zeros((len(Name_Label), len(Name_Feature_Full))),
                               columns=Name_Feature_Full,
                               index=Name_Label)
    for idx in Name_Label:
        dic[idx + '_Feature_Score_mutual'] = abs(mutual_info_regression(Data_Train_Feature_Full, Data_Train_Label[idx]))
        dic[idx + '_Feature_Score_mutual'] =\
            (dic[idx + '_Feature_Score_mutual'] - dic[idx + '_Feature_Score_mutual'].min()) \
            / (dic[idx + '_Feature_Score_mutual'].max() - dic[idx + '_Feature_Score_mutual'].min())

        dic[idx + '_Feature_Score_r'] = abs(r_regression(Data_Train_Feature_Full, Data_Train_Label[idx]))

        dic[idx + '_Feature_Score_AVR'] = (  dic[idx + '_Feature_Score_mutual'].reshape(1, 22) \
                                           + dic[idx + '_Feature_Score_r'].reshape(1, 22)) / 2

        Socre_Table.loc[idx, :] = dic[idx + '_Feature_Score_AVR']

    # Step 2: Select the features with the highest scores
    Name_Feature = []
    for idx in Name_Feature_Full:
        if abs((Socre_Table[idx]).values.max()) >= Feature_Score_Threshold:
            Name_Feature.append(idx)
    Data_Train_Feature = Data_Train_Feature_Full[Name_Feature]
    Data_Test_Feature = Data_Test_Feature_Full[Name_Feature]

    return Name_Feature, Socre_Table, \
           Data_Train_Feature, Data_Test_Feature

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that the prior distribution is not trainable as we fix its parameters.
def Prior(Size_Kernel, Size_Bias, dtype=None):
    n = Size_Kernel + Size_Bias
    Prior_Model = tf.keras.Sequential(
        [tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))]
    )
    return Prior_Model

# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def Posterior(Size_Kernel, Size_Bias, dtype=None):
    n = Size_Kernel + Size_Bias
    Posterior_Model = tf.keras.Sequential(
        [tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
         tfp.layers.MultivariateNormalTriL(n)]
    )
    return Posterior_Model

def negative_loglikelihood(y_real, y_pred):
    return -y_pred.log_prob(y_real)

def Train_DNN(Data_Train_Feature_Smooth, Data_Train_Label_Smooth):
    print('Training...')
    # Parameters
    Name_Feature = Data_Train_Feature_Smooth.columns.tolist()
    Name_Label = Data_Train_Label_Smooth.columns.tolist()

    Hidden_Unit_Structure = Hyper.Hidden_Unit_Structure
    Activation_Type = Hyper.Activation_Type
    Learning_Rate = Hyper.Learning_Rate
    Num_Branch = 1
    Num_Component = Hyper.Num_Component
    Num_Batch = Hyper.Num_Batch
    Num_Epoch = Hyper.Num_Epoch
    Num_Data_Train = len(Data_Train_Feature_Smooth)
    Num_Label = len(Name_Label)

    # Prepare data for training
    Data_Train_TF = tf.data.Dataset.from_tensor_slices((Data_Train_Feature_Smooth, Data_Train_Label_Smooth))
    Data_Train_TF = Data_Train_TF.shuffle(buffer_size=Num_Data_Train).batch(Num_Batch)

    # Create input layer
    Input = tf.keras.Input(shape=(len(Name_Feature),),
                           name='Input_Layer',
                           dtype=tf.float32
                           )

    # Create hidden layers with weight uncertainty using the DenseVariational layer
    Branch_List = []
    # Create hidden layer
    for idx_Branch in range(Num_Branch):
        # The 1st hidden layer
        dic['Output_of_Branch_' + str(idx_Branch)] = tf.keras.layers.Dense(units=Hidden_Unit_Structure[0],
                                                                           activation=Activation_Type,
                                                                           )(Input)

        # The middle:last hidden layers
        for Unit in Hidden_Unit_Structure[1:]:
            dic['Output_of_Branch_' + str(idx_Branch)] = tf.keras.layers.Dense(units=Unit,
                                                                               activation=Activation_Type,
                                                                               )(dic['Output_of_Branch_' + str(idx_Branch)])

        # Save the created branch in a list
        Branch_List.append(dic['Output_of_Branch_' + str(idx_Branch)])

    # Combine the branches as a tree
    Output_of_Tree = tf.keras.layers.concatenate(Branch_List)

    Output = []
    for idx_Label in range(Num_Label):
        # Create output layer
        dic['Output_Point_' + str(idx_Label)] = tf.keras.layers.Dense(units=1,
                                                                      name=Name_Label[idx_Label])(Output_of_Tree)

        Output.append(dic['Output_Point_' + str(idx_Label)])

    # Create NN
    NN = tf.keras.Model(inputs=Input,
                        outputs=Output
                        )

    NN.summary()

    Loss_List = {}
    for idx_Label in range(Num_Label):
        Loss_List[Name_Label[idx_Label]] = 'mse'

    NN.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=Learning_Rate),
               loss=Loss_List)
    Train_History = NN.fit(Data_Train_TF,
                           epochs=Num_Epoch)
    print("Training finished.")
    return NN, Train_History

def Train_BMDN(Data_Train_Feature_Smooth, Data_Train_Label_Smooth):
    print('Training...')
    # Parameters
    Name_Feature = Data_Train_Feature_Smooth.columns.tolist()
    Name_Label = Data_Train_Label_Smooth.columns.tolist()

    Hidden_Unit_Structure = Hyper.Hidden_Unit_Structure
    Activation_Type = Hyper.Activation_Type
    Learning_Rate = Hyper.Learning_Rate
    Num_Branch = Hyper.Num_Branch
    Num_Component = Hyper.Num_Component
    Num_Batch = Hyper.Num_Batch
    Num_Epoch = Hyper.Num_Epoch
    Num_Data_Train = len(Data_Train_Feature_Smooth)
    Num_Label = len(Name_Label)

    Data_Train_Label_Smooth_numpy = Data_Train_Label_Smooth.to_numpy()
    # Prepare data for training
    # Data_Train_TF = tf.data.Dataset.from_tensor_slices((Data_Train_Feature_Smooth, Data_Train_Label_Smooth))
    # Data_Train_TF = Data_Train_TF.shuffle(buffer_size=Num_Data_Train).batch(Num_Batch)

    # Create input layer
    Input = tf.keras.Input(shape=(len(Name_Feature),),
                           name='Input_Layer',
                           dtype=tf.float32
                           )

    # Create hidden layers with weight uncertainty using the DenseVariational layer
    Branch_List = []
    # Create hidden layer
    for idx_Branch in range(Num_Branch):
        # The 1st hidden layer
        dic['Output_of_Branch_' + str(idx_Branch)] = tfp.layers.DenseVariational(units=Hidden_Unit_Structure[0],
                                                                                 make_prior_fn=Prior,
                                                                                 make_posterior_fn=Posterior,
                                                                                 kl_weight=1 / Num_Data_Train,
                                                                                 # kl_use_exact=False,
                                                                                 activation=Activation_Type,
                                                                                 )(Input)

        # The middle:last hidden layers
        for Unit in Hidden_Unit_Structure[1:]:
            dic['Output_of_Branch_' + str(idx_Branch)] = tfp.layers.DenseVariational(units=Unit,
                                                                                     make_prior_fn=Prior,
                                                                                     make_posterior_fn=Posterior,
                                                                                     kl_weight=1 / Num_Data_Train,
                                                                                     # kl_use_exact=False,
                                                                                     activation=Activation_Type,
                                                                                     )(dic['Output_of_Branch_' + str(idx_Branch)])

        # Save the created branch in a list
        Branch_List.append(dic['Output_of_Branch_' + str(idx_Branch)])

    # Combine the branches as a tree
    Output_of_Tree = tf.keras.layers.concatenate(Branch_List)

    Params_Size = tfp.layers.MixtureSameFamily.params_size(
        Num_Component,
        component_params_size=tfp.layers.MultivariateNormalTriL.params_size(Num_Label))

    Distribution_Params = tf.keras.layers.Dense(units=Params_Size)(Output_of_Tree)
    # # this “name” parameter needs to match label_names, since I used dict as container.
    # # output_hour = tfp.layers.MixtureNormal(n_components, [3],  name=label_names[i])(distribution_params)
    # tfpl.IndependentNormal   MultivariateNormalTriL
    Output = tfp.layers.MixtureSameFamily(Num_Component, tfp.layers.MultivariateNormalTriL(Num_Label))(
        Distribution_Params)

    # Create NN
    NN = tf.keras.Model(inputs=Input,
                        outputs=Output
                        )

    NN.summary()

    NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate),
               loss=negative_loglikelihood)
    # changed the data for training
    Train_History = NN.fit(Data_Train_Feature_Smooth,
                           Data_Train_Label_Smooth_numpy,
                           batch_size=Num_Batch,
                           epochs=Num_Epoch)
    print("Training finished.")
    return NN, Train_History

def Predict_and_Evaluate_BMDN(BMDN, Data_Test_Feature, Data_Test_Label, Datetime_Test_List, Col_Max, Col_Min):
    print('Predicting...')
    # Parameters
    Name_Label = Data_Test_Label.columns.tolist()
    Num_Sampling = Hyper.Num_Sampling
    Num_Data_Test = len(Data_Test_Feature)
    Num_Label = len(Name_Label)
    Num_Component = Hyper.Num_Component

    # Step 1: Sampling N times for each testing data point
    # Prepare box for sampling

    Prediction_Sampling_Wise = []
    Prediction_Datapoint_Wise = []
    for idx_Sampling in range(Num_Sampling):
        for idx_Data in range(Num_Data_Test):
            print('Sampling', str(idx_Sampling), 'Data', str(idx_Data))
            Prediction_Datapoint_Wise.append(BMDN(tf.convert_to_tensor(Data_Test_Feature.iloc[[idx_Data]]))) # Predicting
        Prediction_Sampling_Wise.append(Prediction_Datapoint_Wise) # Save the box of this sampling
        Prediction_Datapoint_Wise = [] # Clear the box of this sampling

    # Step 2: Get their confidence
    Prediction_Sampling_Wise_Mean = []
    Prediction_Sampling_Wise_Covar = []
    Prediction_Sampling_Wise_Var = []
    Prediction_Sampling_Wise_Weight = []

    Prediction_Datapoint_Wise_Covar = []
    Prediction_Datapoint_Wise_Mean = []
    Prediction_Datapoint_Wise_Var = []
    Prediction_Datapoint_Wise_Weight = []
    for idx_Sampling in range(Num_Sampling):
        for idx_Data in range(Num_Data_Test):
            Prediction_Datapoint_Wise_Weight.append(Prediction_Sampling_Wise[idx_Sampling][idx_Data].mixture_distribution.probs_parameter().numpy()[0])
            Prediction_Datapoint_Wise_Mean.append(Prediction_Sampling_Wise[idx_Sampling][idx_Data].components_distribution.mean().numpy()[0])
            Prediction_Datapoint_Wise_Covar.append(Prediction_Sampling_Wise[idx_Sampling][idx_Data].components_distribution.covariance().numpy()[0])
            Prediction_Datapoint_Wise_Var.append(Prediction_Sampling_Wise[idx_Sampling][idx_Data].components_distribution.variance().numpy()[0]) # Diagonal elements of the covariance matrix
        # Save the box of this sampling
        Prediction_Sampling_Wise_Weight.append(Prediction_Datapoint_Wise_Weight)
        Prediction_Sampling_Wise_Mean.append(Prediction_Datapoint_Wise_Mean)
        Prediction_Sampling_Wise_Covar.append(Prediction_Datapoint_Wise_Covar)
        Prediction_Sampling_Wise_Var.append(Prediction_Datapoint_Wise_Var)
        # Clear the box of this sampling
        Prediction_Datapoint_Wise_Weight = []
        Prediction_Datapoint_Wise_Mean = []
        Prediction_Datapoint_Wise_Covar = []
        Prediction_Datapoint_Wise_Var = []

    # Calculate confidence
    Prediction_Confidence = np.zeros((Num_Data_Test, Num_Sampling))
    for idx_Sampling in range(Num_Sampling):
        for idx_Data in range(Num_Data_Test):
            Weight1 = Prediction_Sampling_Wise_Weight[idx_Sampling][idx_Data][0]
            Weight2 = Prediction_Sampling_Wise_Weight[idx_Sampling][idx_Data][1]
            for idx_Label in range(len(Name_Label)):
                Var_Vector = np.zeros(len(Name_Label))
                Mean1 = Prediction_Sampling_Wise_Mean[idx_Sampling][idx_Data][0][idx_Label]
                Mean2 = Prediction_Sampling_Wise_Mean[idx_Sampling][idx_Data][1][idx_Label]
                Var1 = Prediction_Sampling_Wise_Var[idx_Sampling][idx_Data][0][idx_Label]
                Var2 = Prediction_Sampling_Wise_Var[idx_Sampling][idx_Data][1][idx_Label]

                Mean = Mean1 * Weight1 + Mean2 * Weight2
                Var = Weight1*Var1 + Weight2*Var2 + Weight1*math.pow(Mean1, 2) + Weight2*math.pow(Mean2, 2) - math.pow(Mean, 2) # pAσ2A+pBσ2B+[pAμ2A+pBμ2B−(pAμA+pBμB)2]
                # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
                Var_Vector[idx_Label] = Var
                # Save the confidence
                Prediction_Confidence[idx_Data][idx_Sampling] = np.linalg.norm(Var_Vector, 1)

    # Step3: Select the GMM with the highest confidence
    idx_Confi_Highest = []
    Prediction_Final_GMM = []
    for idx_Data in range(Num_Data_Test):
        idx_Confi_Highest.append(int(Prediction_Confidence[idx_Data].argmin()))
        idx_Sampling_Selected = idx_Confi_Highest[idx_Data]
        Prediction_Final_GMM.append(Prediction_Sampling_Wise[idx_Sampling_Selected][idx_Data])

    Prediction_Final_GMM_Weight = []
    Prediction_Final_GMM_Mean = []
    Prediction_Final_GMM_Covar =[]
    for idx_Data in range(Num_Data_Test):
        Prediction_Final_GMM_Weight.append(Prediction_Final_GMM[idx_Data].mixture_distribution.probs_parameter().numpy()[0])
        Prediction_Final_GMM_Mean.append(Prediction_Final_GMM[idx_Data].components_distribution.mean().numpy()[0])
        Prediction_Final_GMM_Covar.append(Prediction_Final_GMM[idx_Data].components_distribution.covariance().numpy()[0])
        for idx_Comp in range(Num_Component):
            Prediction_Final_GMM_Covar[idx_Data][idx_Comp] = Prediction_Final_GMM_Covar[idx_Data][idx_Comp]*np.identity(len(Prediction_Final_GMM_Covar[idx_Data][idx_Comp]))

    # Step 4: Reverse normalizations
    Eva_Prediction_Weight = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Component)),
                                         index=Datetime_Test_List['idx'])
    Eva_Prediction_Mean = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                       columns=Name_Label,
                                       index=Datetime_Test_List['idx'])

    Eva_Prediction_Var = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                      columns=Name_Label,
                                      index=Datetime_Test_List['idx'])

    Eva_Prediction_Stdv = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                       columns=Name_Label,
                                       index=Datetime_Test_List['idx'])
    Eva_Prediction_UB = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                     columns=Name_Label,
                                     index=Datetime_Test_List['idx'])
    Eva_Prediction_LB = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                     columns=Name_Label,
                                     index=Datetime_Test_List['idx'])
    Eva_Realization = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                   columns=Name_Label,
                                   index=Datetime_Test_List['idx'])
    for idx_Data in range(Num_Data_Test):
        Weight1 = Prediction_Final_GMM_Weight[idx_Data][0]
        Weight2 = Prediction_Final_GMM_Weight[idx_Data][1]
        # Create box for this component
        Component_Wise_Mean = np.zeros((Num_Label, Num_Component))
        Component_Wise_Covar_Diag = np.zeros((Num_Label, Num_Component))
        for idx_Label in range(Num_Label):
            print(idx_Label)

            # Reverse
            Rea = Data_Test_Label.loc[idx_Data].loc[Name_Label[idx_Label]]*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]]) + Col_Min[Name_Label[idx_Label]]
            Mean1 = Prediction_Final_GMM_Mean[idx_Data][0][idx_Label]*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]]) + Col_Min[Name_Label[idx_Label]]
            Mean2 = Prediction_Final_GMM_Mean[idx_Data][1][idx_Label]*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]]) + Col_Min[Name_Label[idx_Label]]

            Stdv1 = math.sqrt(np.diagonal(Prediction_Final_GMM_Covar[idx_Data][0])[idx_Label])*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]])
            Stdv2 = math.sqrt(np.diagonal(Prediction_Final_GMM_Covar[idx_Data][1])[idx_Label])*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]])

            Var1 = Stdv1 * Stdv1
            Var2 = Stdv2 * Stdv2
            # Save for each component
            Component_Wise_Mean[idx_Label][0] = Mean1
            Component_Wise_Mean[idx_Label][1] = Mean2
            Component_Wise_Covar_Diag[idx_Label][0] = Var1
            Component_Wise_Covar_Diag[idx_Label][1] = Var2
            # SUM
            Mean = Weight1*Mean1 + Weight2*Mean2
            Var = Weight1*Var1 + Weight2*Var2 + Weight1*math.pow(Mean1, 2) + Weight2*math.pow(Mean2, 2) - math.pow(Mean, 2)
            Stdv = math.sqrt(Var)
            # Record
            Eva_Prediction_Weight.loc['Test_Data_' + str(idx_Data)].loc[0] = Weight1
            Eva_Prediction_Weight.loc['Test_Data_' + str(idx_Data)].loc[1] = Weight2
            Eva_Prediction_Mean.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Mean
            Eva_Prediction_Var.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Var
            Eva_Prediction_Stdv.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Stdv
            Eva_Prediction_UB.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Mean + 1.96 * Stdv
            Eva_Prediction_LB.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Mean - 1.96 * Stdv
            Eva_Realization.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Rea
        # Save GMM
        np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Data_" + str(idx_Data+1) + "_GMM_Comp_Mean"       + ".csv", Component_Wise_Mean, delimiter=",")
        np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Data_" + str(idx_Data+1) + "_GMM_Comp_Covar_Diag" + ".csv", Component_Wise_Covar_Diag, delimiter=",")
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Data_All_Comp_GMM_Weight.csv", Eva_Prediction_Weight, delimiter=",")
    # Save All-in-One

    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Prediction_Mean.csv", Eva_Prediction_Mean, delimiter=",")
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Prediction_Stdv.csv", Eva_Prediction_Stdv, delimiter=",")
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Prediction_UB.csv", Eva_Prediction_UB, delimiter=",")
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Prediction_LB.csv", Eva_Prediction_LB, delimiter=",")
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/Matlab/Prediction_Realization.csv", Eva_Realization, delimiter=",")

    #
    Error = Eva_Prediction_Mean - Eva_Realization
    MAE = abs(Error).values.mean()
    MAPE = 100*abs(Error / Eva_Realization).values.mean()
    RMSE = np.sqrt(np.square(abs(Error).values).mean())
    NSE = 1 - sum(sum(np.square(Eva_Realization - Eva_Prediction_Mean).values)) / sum(sum(np.square(Eva_Realization - Eva_Prediction_Mean.mean().mean()).values))

    return Eva_Realization,\
           Eva_Prediction_Mean, Eva_Prediction_Stdv, Eva_Prediction_UB, Eva_Prediction_LB, \
           Error, MAE, MAPE, RMSE, NSE

def Predict_and_Evaluate_DNN(DNN, Data_Test_Feature, Data_Test_Label, Datetime_Test_List, Col_Max, Col_Min):
    print('Predicting...')
    # Parameters
    Name_Label = Data_Test_Label.columns.tolist()
    Num_Data_Test = len(Data_Test_Feature)
    Num_Label = len(Name_Label)

    Prediction_Datapoint_Wise = []
    for idx_Data in range(Num_Data_Test):
        print('Data', str(idx_Data))
        Prediction_Datapoint_Wise.append(DNN(tf.convert_to_tensor(Data_Test_Feature.iloc[[idx_Data]])))

    # Reverse normalizations
    Eva_Prediction_Mean = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                       columns=Name_Label,
                                       index=Datetime_Test_List['idx'])

    Eva_Realization = pd.DataFrame(data=np.zeros((Num_Data_Test, Num_Label)),
                                   columns=Name_Label,
                                   index=Datetime_Test_List['idx'])
    for idx_Data in range(Num_Data_Test):
        for idx_Label in range(Num_Label):
            # Reverse
            Rea = Data_Test_Label.loc[idx_Data].loc[Name_Label[idx_Label]]*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]]) + Col_Min[Name_Label[idx_Label]]
            Mean = Prediction_Datapoint_Wise[idx_Data][idx_Label].numpy()*(Col_Max[Name_Label[idx_Label]] - Col_Min[Name_Label[idx_Label]]) + Col_Min[Name_Label[idx_Label]]
            # Record
            Eva_Prediction_Mean.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Mean
            Eva_Realization.loc['Test_Data_' + str(idx_Data)].loc['Label_' + str(idx_Label)] = Rea
    # Save All-in-One
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/DNN/Prediction_Point.csv", Eva_Prediction_Mean, delimiter=",")
    np.savetxt("/Applications/Project_PyCharm/DOE_CHS_Paper/Results/DNN/Prediction_Realization.csv", Eva_Realization, delimiter=",")
    #
    Error = Eva_Prediction_Mean - Eva_Realization
    MAE = abs(Error).values.mean()
    MAPE = 100*abs(Error / Eva_Realization).values.mean()
    RMSE = np.sqrt(np.square(abs(Error).values).mean())
    NSE = 1 - sum(sum(np.square(Eva_Realization - Eva_Prediction_Mean).values)) / sum(sum(np.square(Eva_Realization - Eva_Prediction_Mean.mean().mean()).values))

    return Eva_Realization,\
           Eva_Prediction_Mean, \
           Error, MAE, MAPE, RMSE, NSE
