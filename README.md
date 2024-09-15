# BMDN

This repository contains the implementation of the BMDN framework for the study presented in our paper titled *"A Carryover Storage Quantification Framework for Mid-Term Cascaded Hydropower Planning: A Portland General Electric System Study."*

## Libraries and Dependencies
Below is the list of all libraries and their versions used in this project:

```
absl-py==1.4.0
astunparse==1.6.3
cachetools==5.3.1
certifi==2023.7.22
charset-normalizer==3.2.0
cloudpickle==2.2.1
contourpy==1.1.0
cycler==0.11.0
dcor==0.6
decorator==5.1.1
dm-tree==0.1.8
emd==0.6.2
flatbuffers==23.5.26
fonttools==4.42.1
gast==0.4.0
google-auth==2.22.0
google-auth-oauthlib==1.0.0
google-pasta==0.2.0
grpcio==1.57.0
h5py==3.9.0
idna==3.4
install==1.3.5
joblib==1.3.2
keras==2.13.1
kiwisolver==1.4.5
libclang==16.0.6
llvmlite==0.40.1
Markdown==3.4.4
MarkupSafe==2.1.3
matplotlib==3.7.2
numba==0.57.1
numpy==1.24.3
oauthlib==3.2.2
opt-einsum==3.3.0
packaging==23.1
pandas==2.0.3
Pillow==10.0.0
protobuf==4.24.2
pyasn1==0.5.0
pyasn1-modules==0.3.0
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3
PyYAML==6.0.1
requests==2.31.0
requests-oauthlib==1.3.1
rsa==4.9
scikit-learn==1.3.0
scipy==1.11.2
seaborn==0.12.2
six==1.16.0
sklearn==0.0
sparse==0.14.0
tabulate==0.9.0
tensorboard==2.13.0
tensorboard-data-server==0.7.1
tensorflow==2.13.0
tensorflow-estimator==2.13.0
tensorflow-io-gcs-filesystem==0.33.0
tensorflow-probability==0.21.0
termcolor==2.3.0
threadpoolctl==3.2.0
typing_extensions==4.5.0
tzdata==2023.3
urllib3==1.26.16
Werkzeug==2.3.7
wrapt==1.15.0
```

## File Overview
- **Hyper.py**: Defines the hyperparameters used in the project.
- **Library.py**: Contains all core functions, including training and testing procedures.
- **Trainer.py**: Calls the functions from *Library.py* to execute the training and testing processes.

## Notes
- The PGE dataset used in this project cannot be made public. However, the provided code should assist in understanding the implementation of the BMDN framework.

If you have any questions, feel free to reach out at xchen130@stevens.edu.
