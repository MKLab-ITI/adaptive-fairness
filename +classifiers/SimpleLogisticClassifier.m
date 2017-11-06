classdef SimpleLogisticClassifier < handle
    properties 
        w
        defaultTrainingRate
        defaultRegularization
        defaultMaxItterations
        defaultConvergence
        trainingErrorTracking
        trackedError
    end
    methods
        function obj = SimpleLogisticClassifier(defaultConvergence, defaultTrainingRate, defaultRegularization, defaultMaxItterations)
            obj.defaultConvergence = 0.001;
            obj.defaultTrainingRate = 0.1;
            obj.defaultRegularization = 0;
            obj.defaultMaxItterations = 10000;
            obj.trainingErrorTracking = false;
            obj.trackedError = [];
            if nargin>= 1
                obj.defaultConvergence = defaultConvergence;
            end
            if nargin>= 2
                obj.defaultTrainingRate = defaultTrainingRate;
            end
            if nargin>= 3
                obj.defaultRegularization = defaultRegularization;
            end
            if nargin>= 4
                obj.defaultMaxItterations = defaultMaxItterations;
            end
        end
        function train(obj, x, y, trainingWeights, previousW, regularization, trainingRate, maxItterations)
            x = [x ones(size(x,1),1)];%add constant term
            if nargin<4
                trainingWeights = ones(length(y),1);
            end
            if nargin < 5
                previousW = zeros(size(x,2),1);
            end
            if nargin < 6
                regularization = obj.defaultRegularization;
            end
            if nargin < 7
                trainingRate = obj.defaultTrainingRate;
            end
            if nargin < 8
                maxItterations = obj.defaultMaxItterations;
            end
            obj.w = previousW;
            xT = x';
            prevError = 1;
            velocities = ones(size(x,2),1);
            %trainingWeightsSum = sum(trainingWeights);
            for itteration=1:maxItterations
                planes = x*obj.w;
                errors = sigmoid(planes)-y;
                error = norm(errors);
                if(abs(error-prevError)<obj.defaultConvergence)
                    break;
                end
                prevError = error;
                derivatives = sigmoidDerivative(planes);
                accumulation = xT*(derivatives.*errors.*trainingWeights)/length(y)+regularization*obj.w/length(y);
                
                velocities = velocities*0.2 + 0.8*(accumulation).^2;
                
                obj.w = obj.w - trainingRate*accumulation./sqrt(velocities+0.1);

                if(obj.trainingErrorTracking)
                    obj.trackedError = [obj.trackedError error/length(y)];
                end
            end
        end
        
        function y = predict(obj, x)
            x = [x ones(size(x,1),1)];%add constant term
            planes = x*obj.w;
            y = sigmoid(planes);
        end
        
        function enableTrainingErrorTracking(obj)
            obj.trainingErrorTracking = true;
        end
    end
end

function s = sigmoid(x)
    s = 1.0./(1.0 + exp(-x));
end
function d = sigmoidDerivative(x)
    d = (exp(-x))./((1+exp(-x)).^2);
end