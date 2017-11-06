classdef AdaptiveWeights < handle
    properties 
        model
        bestParams
        maxItterations
        estimatorType
        continueFromPreviousWeights
    end
    methods
        function obj = AdaptiveWeights(model)
            obj.model = model;
            obj.maxItterations = 12;
            obj.estimatorType = 0;
            obj.continueFromPreviousWeights = false;
        end
        
        function train(obj, x, y, sensitive, objectiveFunction)
            %obj.trainWeights(x, y, sensitive, objectiveFunction);
            options.testflag  = 0;%don't know global min
            options.showits   = 1;%show iterations
            options.maxits  = 80;%max number of iterations
            options.maxevals  = 1200;%max function evaluations
            options.maxdeep   = 200;%max rect divisions
            
            Problem.f = @(params)(-objectiveFunction(obj.trainModel(x, y, sensitive, params, objectiveFunction), x, y, sensitive));
            
            [~,obj.bestParams] = classifiers.Direct(Problem, [0 1;0 1;0 3;0 3], options);
            
            obj.trainModel(x, y, sensitive, obj.bestParams, objectiveFunction);
        end
        
        
        function trainWeights(obj, x, y, sensitive, objectiveFunction)
            trainingSpeedup = 0;
            minReg1 = 0;
            maxReg1 = 1;
            bestReg1 = 0;
            minReg2 = 0;
            maxReg2 = 1;
            bestReg2 = 0;
            bestScore = 10;
            minBeta1 = 0;
            maxBeta1 = 1;
            bestBeta1 =0;
            minBeta2 = 0;
            maxBeta2 = 1;
            bestBeta2 =0;
            regRep1 = 2;
            regRep2 = 2;
            beta1Rep = 2;
            beta2Rep = 2;
            for iter=1:10
                prevScore = bestScore;
                bestScore = -Inf;
                inc1 = (maxReg1-minReg1)/2.0;
                inc2 = (maxReg2-minReg2)/2.0;
                incBeta1 = (maxBeta1-minBeta1)/2.0;
                incBeta2 = (maxBeta2-minBeta2)/2.0;
                trainingSubset = randsample(1:length(y), round((1-trainingSpeedup)*length(y)));
                if(trainingSpeedup>0)
                    trainingValidation = setdiff(1:length(y),trainingSubset);
                else
                    trainingValidation = trainingSubset;
                end
                if(incBeta1==0)
                    beta1Rep = 0;
                end
                if(incBeta2==0)
                    beta2Rep = 0;
                end
                if(inc1==0)
                    regRep1 = 0;
                end
                if(inc2==0)
                    regRep2 = 0;
                end
                for j1=0:beta1Rep
                    beta1 = minBeta1+incBeta1*j1;
                    for j2=0:beta2Rep
                        beta2 = minBeta2+incBeta2*j2;
                        for i1=0:regRep1
                            reg1 = minReg1+inc1*i1;
                            for i2=0:regRep2
                                reg2 = minReg2+inc2*i2;
                                %reg2 = 1-reg1;
                                fprintf('=');
                                obj.trainModel(x(trainingSubset,:),y(trainingSubset),sensitive(trainingSubset),[reg1 reg2 beta1 beta2],objectiveFunction);
                                score = objectiveFunction(obj, x(trainingValidation,:), y(trainingValidation), sensitive(trainingValidation));
                                if(score>bestScore)
                                    bestReg1 = reg1;
                                    bestReg2= reg2;
                                    bestBeta1 = beta1;
                                    bestBeta2 = beta2;
                                    bestScore = score;
                                end
                            end
                        end
                    end
                end

                minReg1 = max(0,bestReg1-inc1/2);
                maxReg1 = min(1,bestReg1+inc1/2);
                minReg2 = max(0,bestReg2-inc2/2);
                maxReg2 = min(1,bestReg2+inc2/2);
                if(bestBeta1==minBeta1 || bestBeta1==maxBeta1)
                    incBeta1 = incBeta1*2;
                end
                if(bestBeta2==minBeta2 || bestBeta2==maxBeta2)
                    incBeta2 = incBeta2*2;
                end
                minBeta1 = max(0,bestBeta1-incBeta1/2);%TODO: /2 is forgotten
                maxBeta1 = bestBeta1+incBeta1/2;
                minBeta2 = max(0,bestBeta2-incBeta2/2);%TODO: /2 is forgotten
                maxBeta2 = bestBeta2+incBeta2/2;
        %         if(abs(minBeta1-maxBeta1)<0.1)
        %             maxBeta1 = bestBeta1;
        %             minBeta1 = maxBeta1;
        %         end
        %         if(abs(minBeta2-maxBeta2)<0.1)
        %             maxBeta2 = bestBeta2;
        %             minBeta2 = maxBeta2;
        %         end
                fprintf('\nBernouli = %g %g   Lipshitz = %g, %g   Objective %g\n', bestReg1, bestReg2, bestBeta1, bestBeta2, bestScore);
                if(abs(bestScore-prevScore)<0.005)
                    %break;
                end
                obj.bestParams = [bestReg1 bestReg2 bestBeta1 bestBeta2];
            end
            obj.bestParams = [bestReg1 bestReg2 bestBeta1 bestBeta2];
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
            prevObjective = -Inf;
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
