function [x, y, sensitive, training, test] = sameSignMistreatment(L)
    if nargin<1
        L = 2500;
    end
    x = [mvnrnd([1, 2], [5, 2; 2, 5], L);  mvnrnd([2, 3], [10, 1; 1, 4], L); ...
         mvnrnd([0, -1], [7, 1; 1, 7], L);  mvnrnd([-5, 0], [5, 1; 1, 5], L)...
        ];
    sensitive = [zeros(L,1); ones(L,1); zeros(L,1); ones(L,1)]==1;
    y = [ones(L,1); ones(L,1); zeros(L,1); zeros(L,1)];

    training = randsample(1:length(y), 2*L);
    test = setdiff(1:length(y), training);
end