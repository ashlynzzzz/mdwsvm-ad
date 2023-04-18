# mdwsvm-ad
EECS 553 Course Research Project (2023.02-2023.04)
Angle-based Multicategory Distance-weighted SVM with Anomaly Detector

This project mainly explores one support vector machine (SVM) method and extends it for anomaly detection. An angle-based multicategory distance-weighted SVM model (MDWSVM) proposed by Sun et al. (2017) is first reimplemented, and our results comfirm that MDWSVM outperforms angled-based multicategory SVM (MSVM) and multicategory distance-weighted discrimination (MDWD) with unbalanced datasets. Then MDWSVM is merged with one-class SVM to get a new model called MDWSVM-AD for anomaly detection. Experimental results demonstrate its success in identifying anomaly class.

EMNIST dataset used in this project is available at https://www.nist.gov/itl/iad/image-group/emnist-dataset.