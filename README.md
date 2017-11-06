# AdaptiveFairness
This is a MATLAB project which utilizes adaptive sensitive reweighting to produce fair classifications.

## Classifiers
In the folder *+classifiers* we include our fairness-aware method, alongside some similar experimental approache.

## Datasets
In the folder *+dataImport* we have collect the Adult, Bank and COMPAS datasets and two synthetic disparate mistreatment datasets.
Those datasets can be imported using the respective functions from this folder (e.g. *dataImport.importAdultData()*)
which construct features, labels, sensitive group information and an example split between training and validation samples.

