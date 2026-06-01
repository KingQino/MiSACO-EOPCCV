classdef MiSACO < ALGORITHM
% <single> <real>  <expensive>
% Multisurrogate-Assisted Ant Colony Optimization
% Np --- 60 --- Size of parent population
% No --- 100 --- Size of offspring population

%------------------------------- Reference --------------------------------
% Liu J, Wang Y, Sun G, et al, Multisurrogate-assisted ant colony 
% optimization for expensive optimization problems with continuous and 
% categorical variables, IEEE Transactions on Cybernetics, 2021, 
% 52(11): 11348-11361.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

	methods
        function main(Algorithm,Problem)
            [History,EvalLog,SurLog] = InitializeLogs();
            currentBest = inf;
            runStart    = tic;
            algorithmFile = which(class(Algorithm));
            historyFile = getenv('MISACO_HISTORY_FILE');
            if isempty(historyFile)
                historyFile = fullfile(fileparts(fileparts(algorithmFile)),'history.mat');
            end

            try
                %% Generate random population
                [Np,No] = Algorithm.ParameterSet(60,100);
                PI      = UniformPoint(Np,Problem.D,'Latin');
                feBefore = Problem.FE;
                DB     = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Np,1).*PI+repmat(Problem.lower,Np,1));
                [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,DB,'Init',feBefore,currentBest);
                P.decs = DB.decs;
                P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                P.objs = DB.objs;
                
                while Algorithm.NotTerminated(DB)
                    %% Multisurrogate-Assisted Selection
                    % Surrogate Construction
                    Train_X = DB.decs;
                    Train_X(:,Problem.Dr+1:Problem.D) = ceil(Train_X(:,Problem.Dr+1:Problem.D));
                    Train_Y = DB.objs;
                    upper_r = Problem.upper; upper_r = upper_r(:,1:Problem.Dr);
                    lower_r = Problem.lower; lower_r = lower_r(:,1:Problem.Dr);
                    RBF_glo = RBFCreate_Hamming(Train_X,Train_Y,Problem.Dc,upper_r,lower_r,'gaussian');
                    RF_net  = fitrensemble(Train_X,Train_Y,'CategoricalPredictors',[Problem.Dr+1:Problem.D]);
                    
                    % Generate Offspring Population
                    [~,rank_v] = sort(P.objs); [~,rank_v] = sort(rank_v);
                    OffDec = ACO_MV_generate(P.decs,Problem.Dr,Problem.Dc,Np,Problem.len_c,rank_v,No,upper_r,lower_r);
                    
                    % RBF-Based Selection,
                    RBF_evaluate = zeros(No,1);
                    for i = 1 : No
                        RBF_evaluate(i) = RBFInterp_Hamming(OffDec(i,:),Problem.Dc, upper_r, lower_r, RBF_glo);
                    end
                    finiteRBF = isfinite(RBF_evaluate);
                    if any(finiteRBF)
                        RBF_rank_value = RBF_evaluate;
                        RBF_rank_value(~finiteRBF) = inf;
                        [~,rank] = sort(RBF_rank_value);
                        id_selected = rank(1);
                        pred_rbf    = RBF_evaluate(id_selected);
                    else
                        id_selected = randi(size(OffDec,1));
                        pred_rbf    = nan;
                    end
                    feBefore    = Problem.FE;
                    New         = Problem.Evaluation(OffDec(id_selected,:));
                    SurLog      = LogSurrogatePrediction(SurLog,'RBF',pred_rbf,New.objs,Problem.FE);
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RBF',feBefore,currentBest);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];
                    
                    % LSBT-Based Selection
                    RF_evaluate = predict(RF_net,OffDec);
                    finiteRF = isfinite(RF_evaluate);
                    if any(finiteRF)
                        RF_rank_value = RF_evaluate;
                        RF_rank_value(~finiteRF) = inf;
                        [~,rank] = sort(RF_rank_value);
                        id_selected = rank(1);
                        pred_rf     = RF_evaluate(id_selected);
                    else
                        id_selected = randi(size(OffDec,1));
                        pred_rf     = nan;
                    end
                    feBefore    = Problem.FE;
                    New         = Problem.Evaluation(OffDec(id_selected,:));
                    SurLog      = LogSurrogatePrediction(SurLog,'RF',pred_rf,New.objs,Problem.FE);
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RF',feBefore,currentBest);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];
                    
                    % Random Selection
                    id_selected = randi(size(OffDec,1));
                    feBefore    = Problem.FE;
                    New         = Problem.Evaluation(OffDec(id_selected,:));
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'Random',feBefore,currentBest);
                    DB = [DB, New];
                    
                    %% Surrogate-Assisted Local Search
                    [~,id_best] = min(DB.objs);
                    if length(id_best) > 1
                        id_best = id_best(1,:);
                    end
                    best_x      = DB(id_best).decs;
                    best_r      = best_x(:,1:Problem.Dr);
                    best_c      = ceil(best_x(:,Problem.Dr+1:Problem.D));
                     
                    Train_x     = DB.decs;
                    idx_local  = dist_ham(best_c,ceil(Train_x(:,Problem.Dr+1:Problem.D)));
                    best_Arc_r = Train_x(idx_local,1:Problem.Dr);
                    best_Arc_f = DB(idx_local).objs;
                    
                    Local_flag = sum(idx_local)> 5*Problem.Dr;
                    if Local_flag
                        idx_rank_best = dist_r(best_r,best_Arc_r);
                        local_up = max(best_Arc_r(idx_rank_best,:));
                        local_dn = min(best_Arc_r(idx_rank_best,:));
                        [xr_Local,local_success] = Local_Search(best_Arc_r(idx_rank_best,:),best_Arc_f(idx_rank_best),local_up,local_dn,best_r);
                        if local_success
                            feBefore = Problem.FE;
                            New      = Problem.Evaluation([xr_Local,best_c]);
                            [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'LocalSearch',feBefore,currentBest);
                            DB = [DB, New];
                        end
                    end
                    
                    %% Population Update
                    [~,rank] = sort(DB.objs);
                    P.decs = DB(rank(1:Np)).decs;
                    P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                    P.objs = DB(rank(1:Np)).objs;
                end
            catch err
                FinalizeLogs(History,EvalLog,SurLog,runStart,Problem,historyFile);
                rethrow(err);
            end
            FinalizeLogs(History,EvalLog,SurLog,runStart,Problem,historyFile);
        end
	end
end

function idx_sel = dist_ham(x,v)
    distt = mean(x~=v,2);
    idx_sel = (distt == 0);
end

function out = dist_r(x0,x)
    dist = sqrt(sum((x0 - x).^2,2));
    [~,sort_rank] = sort(dist);
    [~,out_sort] = sort(sort_rank);
    out = (out_sort<=30);
end

function [History,EvalLog,SurLog] = InitializeLogs()
    History.FE   = [];
    History.Best = [];

    EvalLog.FE       = [];
    EvalLog.Source   = {};
    EvalLog.Obj      = [];
    EvalLog.Best     = [];
    EvalLog.Improved = [];

    SurLog.FE     = [];
    SurLog.Source = {};
    SurLog.Pred   = [];
    SurLog.Real   = [];
end

function [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,Population,source,feBefore,currentBest)
    objs = Population.objs;
    objs = objs(:,1);
    for i = 1 : size(objs,1)
        newObj        = objs(i);
        isImprovement = newObj < currentBest;
        currentBest   = min(currentBest,newObj);
        fe            = feBefore + i;

        History.FE(end+1,1)   = fe;
        History.Best(end+1,1) = currentBest;

        EvalLog.FE(end+1,1)       = fe;
        EvalLog.Source{end+1,1}   = source;
        EvalLog.Obj(end+1,1)      = newObj;
        EvalLog.Best(end+1,1)     = currentBest;
        EvalLog.Improved(end+1,1) = isImprovement;
    end
end

function SurLog = LogSurrogatePrediction(SurLog,source,predicted,realValue,fe)
    SurLog.FE(end+1,1)     = fe;
    SurLog.Source{end+1,1} = source;
    SurLog.Pred(end+1,1)   = predicted(1);
    SurLog.Real(end+1,1)   = realValue(1);
end

function RunSummary = FinalizeLogs(History,EvalLog,SurLog,runStart,Problem,historyFile)
    RunSummary = BuildRunSummary(History,EvalLog,SurLog,runStart,Problem);
    save(historyFile,'History','EvalLog','SurLog','RunSummary');
    PrintRunSummary(RunSummary,historyFile);
end

function RunSummary = BuildRunSummary(History,EvalLog,SurLog,runStart,Problem)
    sources = {'RBF','RF','Random','LocalSearch'};

    if isempty(EvalLog.Obj)
        bestObj = nan;
        bestFE  = nan;
    else
        [bestObj,idBest] = min(EvalLog.Obj);
        bestFE = EvalLog.FE(idBest);
    end

    RunSummary.BestObjective = bestObj;
    RunSummary.BestFE        = bestFE;
    RunSummary.Runtime       = toc(runStart);
    RunSummary.TotalFE       = Problem.FE;

    for i = 1 : length(sources)
        source = sources{i};
        RunSummary.([source,'Evaluations'])  = CountSource(EvalLog,source);
        RunSummary.([source,'Improvements']) = CountImprovement(EvalLog,source);
    end

    [RunSummary.RBF_MAE,RunSummary.RBF_RMSE] = SurrogateError(SurLog,'RBF');
    [RunSummary.RF_MAE, RunSummary.RF_RMSE]  = SurrogateError(SurLog,'RF');
end

function count = CountSource(EvalLog,source)
    count = sum(strcmp(EvalLog.Source,source));
end

function count = CountImprovement(EvalLog,source)
    count = sum(strcmp(EvalLog.Source,source) & logical(EvalLog.Improved));
end

function [mae,rmse] = SurrogateError(SurLog,source)
    idx = strcmp(SurLog.Source,source);
    if any(idx)
        err  = SurLog.Pred(idx) - SurLog.Real(idx);
        err  = err(isfinite(err));
        if isempty(err)
            mae  = nan;
            rmse = nan;
        else
            mae  = mean(abs(err));
            rmse = sqrt(mean(err.^2));
        end
    else
        mae  = nan;
        rmse = nan;
    end
end

function PrintRunSummary(RunSummary,historyFile)
    fprintf('\n========== MiSACO Run Summary ==========\n');
    fprintf('Best Objective: %.12g\n',RunSummary.BestObjective);
    fprintf('Best FE: %d\n',RunSummary.BestFE);
    fprintf('Runtime: %.2fs\n',RunSummary.Runtime);
    fprintf('Total FE: %d\n\n',RunSummary.TotalFE);

    fprintf('RBF Evaluations: %d\n',RunSummary.RBFEvaluations);
    fprintf('RF Evaluations: %d\n',RunSummary.RFEvaluations);
    fprintf('Random Evaluations: %d\n',RunSummary.RandomEvaluations);
    fprintf('LocalSearch Evaluations: %d\n\n',RunSummary.LocalSearchEvaluations);

    fprintf('RBF Improvements: %d\n',RunSummary.RBFImprovements);
    fprintf('RF Improvements: %d\n',RunSummary.RFImprovements);
    fprintf('Random Improvements: %d\n',RunSummary.RandomImprovements);
    fprintf('LocalSearch Improvements: %d\n\n',RunSummary.LocalSearchImprovements);

    fprintf('RBF MAE: %.12g, RBF RMSE: %.12g\n',RunSummary.RBF_MAE,RunSummary.RBF_RMSE);
    fprintf('RF MAE: %.12g, RF RMSE: %.12g\n',RunSummary.RF_MAE,RunSummary.RF_RMSE);
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('========================================\n\n');
end
