function bestParams = HeuristicDirect(problem, options)
    minReg1 = 0;
    maxReg1 = 1;
    bestReg1 = 0;
    minReg2 = 0;
    maxReg2 = 1;
    bestReg2 = 0;
    minBeta1 = 0;
    maxBeta1 = 3;
    bestBeta1 =0;
    minBeta2 = 0;
    maxBeta2 = 3;
    bestBeta2 =0;
    regRep1 = 2;
    regRep2 = 2;
    beta1Rep = 2;
    beta2Rep = 2;
    prevSigma = Inf;
    
    map = containers.Map;
    
    for iter=1:options.maxits
        scoreSum = 0;
        scoreSquareSum = 0;
        scoreNum = 0;
        bestScore = Inf;
        inc1 = (maxReg1-minReg1)/2.0;
        inc2 = (maxReg2-minReg2)/2.0;
        incBeta1 = (maxBeta1-minBeta1)/2.0;
        incBeta2 = (maxBeta2-minBeta2)/2.0;
       
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
                        fprintf('=');
                        params = [reg1 reg2 beta1 beta2];
%                         id = mat2str(params);
%                         if map.isKey(id)
%                             score = map(id);
%                             id
%                         else
%                             score = problem(params);
%                             map(id) = score;
%                         end
                        score = problem(params);
                        scoreSum = scoreSum+score;
                        scoreSquareSum = scoreSquareSum + score^2;
                        scoreNum = scoreNum+1;
                        if(score<bestScore)
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
        minBeta1 = max(0,bestBeta1-incBeta1/2);
        maxBeta1 = bestBeta1+incBeta1/2;
        minBeta2 = max(0,bestBeta2-incBeta2/2);
        maxBeta2 = bestBeta2+incBeta2/2;
        scoreSigma = sqrt(scoreSquareSum/scoreNum - (scoreSum/scoreNum)^2);
        fprintf('\nIter. %d   Bernouli = %g %g   Lipshitz = %g, %g   Objective %g +-%g\n', iter,  bestReg1, bestReg2, bestBeta1, bestBeta2, bestScore, scoreSigma);
        
        
        %bestParams = [bestReg1 bestReg2 bestBeta1 bestBeta2];
        if(scoreSigma<0.001)
            break;
        end
        prevSigma = scoreSigma;
    end
    bestParams = [bestReg1 bestReg2 bestBeta1 bestBeta2];
end
