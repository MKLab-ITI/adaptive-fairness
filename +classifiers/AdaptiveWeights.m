classdef AdaptiveWeights < handle
    properties 
        model
        bestParams
        maxItterations
        estimatorType
        continueFromPreviousWeights
        heuristicTraining
    end
    methods
        function obj = AdaptiveWeights(model, heuristicTraining)
            if nargin<2
                heuristicTraining = false;
            end
            obj.model = model;
            obj.maxItterations = 12;
            obj.estimatorType = 0;
            obj.continueFromPreviousWeights = true;
            obj.heuristicTraining = heuristicTraining;
        end
        
        function train(obj, x, y, sensitive, objectiveFunction)
            options.testflag  = 0;%don't know global min
            options.showits   = 1;%show iterations
            options.maxits  = 80;%max number of iterations
            options.maxevals  = 1200;%max function evaluations
            options.maxdeep   = 200;%max rect divisions
            
            directLoss = @(params)(-objectiveFunction(obj.trainModel(x, y, sensitive, params, objectiveFunction), x, y, sensitive));
            
            if(obj.heuristicTraining)
                obj.bestParams = classifiers.HeuristicDirect(directLoss, options);
            else
                Problem.f = directLoss;
                [~,obj.bestParams] = classifiers.Direct(Problem, [0 1;0 1;0 3;0 3], options);
            end
            obj.trainModel(x, y, sensitive, obj.bestParams, objectiveFunction);
        end
        
        function obj = trainModel(obj, x, y, sensitive, parameters, objectiveFunction, showConvergence)
            if nargin<7
                showConvergence = false;
            end
            mislabelBernoulliMean(1) = parameters(1);
            mislabelBernoulliMean(2) = parameters(2);
            convexity(1) = parameters(3);
            convexity(2) = parameters(4);
            convergence = [];
            objective = [];
            nonSensitive = ~sensitive;

            if(sum(y(sensitive))/sum(sensitive)<sum(y(nonSensitive))/sum(nonSensitive))
                tmp = sensitive;
                sensitive = nonSensitive;
                nonSensitive = tmp;
            end

            trainingWeights = ones(length(y),1);
            repeatContinue = 1;
            itteration = 0;
            prevObjective = Inf;
            while(itteration<obj.maxItterations && repeatContinue>0.01)
                itteration = itteration+1;
                prevWeights = trainingWeights;
                obj.model.train(x, y, trainingWeights);
                scores = obj.model.predict(x);
                trainingWeights(sensitive) ...
                    = obj.convexLoss(scores(sensitive)-y(sensitive),convexity(1))*mislabelBernoulliMean(1) ... 
                    + obj.convexLoss(y(sensitive)-scores(sensitive),convexity(1))*(1-mislabelBernoulliMean(1));
                trainingWeights(nonSensitive) ...
                    = obj.convexLoss(scores(nonSensitive)-y(nonSensitive),convexity(2))*(1-mislabelBernoulliMean(2)) ... 
                    + obj.convexLoss(y(nonSensitive)-scores(nonSensitive),convexity(2))*(mislabelBernoulliMean(2));

                trainingWeights = trainingWeights/sum(trainingWeights)*length(trainingWeights);
                repeatContinue = norm(trainingWeights-prevWeights);
                
                objective = objectiveFunction(obj, x, y, sensitive);
                if(objective<prevObjective && itteration>obj.maxItterations-2)
                    trainingWeights = prevWeights;
                    obj.model.train(x, y, trainingWeights);
                    break;
                end
                prevObjective = objective;
                if(iscell(showConvergence))
                    convergence = [convergence sqrt(sum((trainingWeights-prevWeights).^2)/length(trainingWeights))];
                    objective = [objective objective];
                end
            end
            %fprintf('finished within %d itterations for tradeoff %f\n', itteration, tradeoff);
            if(iscell(showConvergence))
                figure(1);
                hold on
                plot(1:length(convergence),convergence,showConvergence{1});
                xlabel('Iteration')
                ylabel('Root Mean Square of Weight Edits')
                figure(2);
                hold on
                plot(1:obj.maxItterations,[objective ones(1,obj.maxItterations-length(objective))*objective(end)],showConvergence{1});
                xlabel('Iteration')
                ylabel('Objective')
            end
        end
        
        function y = predict(obj, x)
            y = obj.model.predict(x);
        end
        
        function L = convexLoss(obj, p, beta)
            if(nargin<3)
                beta = 1;
            end
            if(obj.estimatorType==0)
                L = exp(p*beta);
            else
                error('Invalid CULEP estimator type');
            end
        end
    end
end
