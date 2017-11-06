function [x, y, sensitive, training, test] = importAdultData()
    incomeData = dataset('File', '+dataImport/adult.data','ReadVarNames',false,'Delimiter',',');
    incomeDataTest = dataset('File', '+dataImport/adult.test','ReadVarNames',false,'Delimiter',',');
    testSplitPoint = size(incomeData,1);
    incomeData = cat(1,incomeData,incomeDataTest);

    y = ones(size(incomeData,1),1);
    for i=1:length(y)
        if(strcmp(cellstr(incomeData(i,15)),'<=50K')==1)
            y(i) = 0;
        elseif(strcmp(cellstr(incomeData(i,15)),'<=50K.')==1)
            y(i) = 0;
        end
    end
    
    sensitive = strcmp(cellstr(incomeData(:,10)),'Female')==1;%females are sensitive
    %sensitive = strcmp(cellstr(incomeData(:,9)),'White')==0;%non-whites are sensitive
    
    x =[dataImport.convertToDouble(incomeData(:,1)) dataImport.convertToValues(incomeData(:,2)) dataImport.convertToDouble(incomeData(:,3)) ...
        dataImport.convertToValues(incomeData(:,4)) dataImport.convertToDouble(incomeData(:,5)) dataImport.convertToValues(incomeData(:,6)) ...
        dataImport.convertToValues(incomeData(:,7)) dataImport.convertToValues(incomeData(:,8)) dataImport.convertToValues(incomeData(:,9)) ...
        dataImport.convertToValues(incomeData(:,10)) dataImport.convertToDouble(incomeData(:,11)) dataImport.convertToDouble(incomeData(:,12)) ...
        dataImport.convertToDouble(incomeData(:,13)) dataImport.convertToValues(incomeData(:,14))];
    
    %% generate training and test data
    training = 1:testSplitPoint;
    test = (testSplitPoint+1):length(y);
end
