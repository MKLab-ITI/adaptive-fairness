function x = convertToValues(data)
    %% convert to integer values
    data = dataset2cell(data);
    classes = unique(data);
    mapObj = containers.Map(classes,1:size(classes,1));
    for i=2:size(data,1)
         x(i-1) = mapObj(char(data(i)));
    end
    %% convert to binary
    x = de2bi(x);
end

