# Data Imputation

Data imputation for visualisation

This repository contains two function which can be used for data imputation before data visualisation by elastic graphs, elastic maps or any other techniques.

Function <b>degup</b> calculates fraction of missing data in each record and each feature and then remove the record or feature with greatest fraction of missing data. This procedure is repeated until there is records or features with missed data.

Function <b>kNNImpute</b> imputes data by weighted mean of k nearest neighbour. Nearest neighbours are defined by known values and intervals of distribution of unknown values.

Function <b>svdWithGaps</b> imputes data by decomposition of data matrix 'data' into singular vectors and later reconstruct all values.

## Reference

Please refer as
Mirkes, EM, Zinovyev A, Gorban, AN “Data imputation”, available online on https://github.com/Mirkes/DataImputation, accessed Date_of_Access

## Acknowledgements

Supported by the University of Leicester (UK), Institut Curie (FR), the Ministry of Education and Science of the Russian Federation, project № 14.Y26.31.0022
