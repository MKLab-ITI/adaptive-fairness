[x, y, sensitive, training, test] = dataImport.importCompassData();

folds = 5;

accs = 0;
pRules = 0;
DFPRs = 0;
DFNRs = 0;
for fold=1:folds
    classifier = classifiers.SimpleLogisticClassifier(0.0001);
    if(folds~=1)
        training = randsample(1:length(y), round(length(training)));
        test = setdiff(1:length(y), training);
    end
    classifier.train(x(training,:),y(training));
    [~, acc, AUC, pRule, DFPR, DFNR] = obtainMetrics(classifier, x(test,:), y(test), sensitive(test));
    accs = accs+acc/folds;
    pRules = pRules+pRule/folds;
    DFPRs = DFPRs+DFPR/folds;
    DFNRs = DFNRs+DFNR/folds;
    fprintf('\nCurrent evaluation: acc = %f , pRule = %f , DFPR = %f , DFNR = %f \n\n\n', accs*folds/fold, pRules*folds/fold, DFPRs*folds/fold, DFNRs*folds/fold); 
end