function x = convertToDouble(val)
    x = double(val);
    if(min(x)==-1)
        nx = (x==-1);
        x = [nx, x.*(1-nx)/mean(x.*(1-nx))];
    else
        x = x/mean(x);
    end
   % x = x/std(x);
end