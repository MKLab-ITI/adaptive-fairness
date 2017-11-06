function [x, y, sensitive, training, test] = importBankData()
   data = dataset('File', '+dataImport/bank-full.csv','ReadVarNames',true,'Delimiter',';');
   %data = dataset('File', '+dataImport/bank.csv','ReadVarNames',true,'Delimiter',';');
   % testSplitPoint = size(data,1);
   % dataTest = dataset('File', 'bank.csv','ReadVarNames',true,'Delimiter',';');
   % data = cat(1,data,dataTest);

    y = ones(size(data,1),1);
    for i=1:length(y)
        if(strcmp(cellstr(data(i,17)),'no')==1)
            y(i) = 0;
        end
    end
    
    sensitive = double(data(:,1))>=25 & double(data(:,1))<=60;
    
    x =[dataImport.convertToDouble(data(:,1)) dataImport.convertToValues(data(:,2)) dataImport.convertToValues(data(:,3)) ...
        dataImport.convertToValues(data(:,4)) dataImport.convertToValues(data(:,5)) dataImport.convertToDouble(data(:,6)) ...
        dataImport.convertToValues(data(:,7)) dataImport.convertToValues(data(:,8)) dataImport.convertToValues(data(:,9)) ...
        dataImport.convertToDouble(data(:,10)) dataImport.convertToValues(data(:,11)) dataImport.convertToDouble(data(:,12)) ...
        dataImport.convertToDouble(data(:,13)) dataImport.convertToDouble(data(:,14)) dataImport.convertToDouble(data(:,15)) ...
        dataImport.convertToValues(data(:,16))];

    %% generate training and test data
    training = randsample(1:length(y), round(length(y)*0.7));
    test = setdiff(1:length(y), training);
end
