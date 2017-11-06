[x, y, sensitive, training, test] = dataImport.sameSignMistreatment();
folds = 5;
x = [x sensitive];

classifier = classifiers.AdaptiveWeights(classifiers.SimpleLogisticClassifier(0.001));

accs = 0;
pRules = 0;
DFPRs = 0;
DFNRs = 0;

validationFunction = @(c,x,y,s)obtainMetrics(c,x,y,s,[2 0 0 -1 -1]);

for fold=1:folds
    if(folds~=1)
        training = randsample(1:length(y), length(training));
        test = setdiff(1:length(y), training);
    end
    classifier.train(x(training,:),y(training),sensitive(training),validationFunction);
    [~, acc, ~, pRule, DFPR, DFNR] = validationFunction(classifier,x(training,:),y(training),sensitive(training));
    accs = accs+acc/folds;
    pRules = pRules+pRule/folds;
    DFPRs = DFPRs+DFPR/folds;
    DFNRs = DFNRs+DFNR/folds;
    fprintf('\nCurrent evaluation: acc = %f , pRule = %f , DFPR = %f , DFNR = %f \n\n\n', accs*folds/fold, pRules*folds/fold, DFPRs*folds/fold, DFNRs*folds/fold); 
end