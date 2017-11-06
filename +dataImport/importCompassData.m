function [x, y, sensitive, training, test] = importCompassData()
    % conn = sqlite('compas.db','readonly');
    % data = fetch(conn,'SELECT * FROM compas');
    % close(conn)
    % clear conn

    data = dataset('File', '+dataImport/compas-scores-two-years.csv','ReadVarNames',false,'Delimiter',',');
    data = data(strcmp(cellstr(data(:,10)),'Caucasian')|strcmp(cellstr(data(:,10)),'African-American'),:);
    data = data(strcmp(cellstr(data(:,23)),'O')==0,:);
    
    tmp = str2double(dataset2cell(data(:,15)));
    tmp = tmp(2:end);
    x =[dataImport.convertToValues(data(:,10)) dataImport.convertToValues(data(:,6)) dataImport.convertToValues(data(:,9)) ...
        tmp/mean(tmp) dataImport.convertToValues(data(:,23))];
    
    y = str2double(dataset2cell(data(:,53)));
    y = y(2:end);

    sensitive = strcmp(cellstr(data(:,10)),'Caucasian')==0;
    
    training = 1:floor(size(data,1)*0.5);
    test = (length(training)+1):length(y);
end