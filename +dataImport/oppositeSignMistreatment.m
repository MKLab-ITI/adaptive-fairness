function [x, y, sensitive, training, test] = oppositeSignMistreatment(L)
    if nargin<1
        L = 2500;
    end

    x = [mvnrnd([2 0], [5, 1; 1, 5], L);  mvnrnd([2, 3], [5, 1; 1, 5], L); ...
         mvnrnd([-1, -3], [5, 1; 1, 5], L);  mvnrnd([-1, 0], [5, 1; 1, 5], L)...
        ];
    sensitive = [zeros(L,1); ones(L,1); zeros(L,1); ones(L,1)]==1;
    y = [ones(L,1); ones(L,1); zeros(L,1); zeros(L,1)];
    
    training = randsample(1:length(y), 2*L);
    test = setdiff(1:length(y), training);
end