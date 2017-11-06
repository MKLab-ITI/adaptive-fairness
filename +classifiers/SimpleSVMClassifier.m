classdef SimpleSVMClassifier < handle
    properties 
        model
    end
    methods
        function train(obj, x, y, trainingWeights)
            if nargin<4
                trainingWeights = ones(length(y),1);
            end
            obj.model = fitcsvm (x, y, 'Weights', trainingWeights);
        end
        
        function y = predict(obj, x)
            [label,score] = predict(obj.model,x);
            y = label.*abs(score(1)) + (1-label).*(1-abs(score(1)));
        end
    end
end
