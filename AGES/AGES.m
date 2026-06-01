classdef AGES < ALGORITHM
% <single> <real> <expensive>
% Alignment-Guided Candidate Selection for EOPCCV.
% Np       --- 60   --- Size of parent population
% No       --- 100  --- Size of offspring population
% TopK     --- 20   --- Candidate shortlist size for alignment scoring
% MinAlign --- 20   --- Minimum alignment samples before using the model
% Lambda   --- 0.5  --- Penalty weight for predicted surrogate error
% Beta     --- 0.15 --- Reward weight for novelty

    methods
        function main(Algorithm,Problem)
            [History,EvalLog,SurLog,AlignLog] = InitializeLogs();
            AlignDB     = InitializeAlignmentDB();
            currentBest = inf;
            runStart    = tic;
            algorithmFile = which(class(Algorithm));
            historyFile = getenv('AGES_HISTORY_FILE');
            if isempty(historyFile)
                historyFile = getenv('MISACO_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = fullfile(fileparts(fileparts(algorithmFile)),'history.AGES.mat');
            end

            try
                [Np,No,TopK,MinAlign,Lambda,Beta] = Algorithm.ParameterSet(60,100,20,20,0.5,0.15);
                TopK     = max(1,round(TopK));
                MinAlign = max(0,round(MinAlign));

                PI       = UniformPoint(Np,Problem.D,'Latin');
                feBefore = Problem.FE;
                DB       = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Np,1).*PI+repmat(Problem.lower,Np,1));
                [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,DB,'Init',feBefore,currentBest);

                P.decs = DB.decs;
                P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                P.objs = DB.objs;

                while Algorithm.NotTerminated(DB)
                    Train_X = DB.decs;
                    Train_X(:,Problem.Dr+1:Problem.D) = ceil(Train_X(:,Problem.Dr+1:Problem.D));
                    Train_Y = DB.objs;
                    upper_r = Problem.upper(:,1:Problem.Dr);
                    lower_r = Problem.lower(:,1:Problem.Dr);

                    RBF_glo = RBFCreate_Hamming(Train_X,Train_Y,Problem.Dc,upper_r,lower_r,'gaussian');
                    RF_net  = fitrensemble(Train_X,Train_Y,'CategoricalPredictors',Problem.Dr+1:Problem.D);

                    [~,rank_v] = sort(P.objs);
                    [~,rank_v] = sort(rank_v);
                    OffDec = ACO_MV_generate(P.decs,Problem.Dr,Problem.Dc,Np,NormalizeLenC(Problem.len_c,Problem.Dc),rank_v,No,upper_r,lower_r);

                    RBF_evaluate = zeros(size(OffDec,1),1);
                    for i = 1 : size(OffDec,1)
                        RBF_evaluate(i) = RBFInterp_Hamming(OffDec(i,:),Problem.Dc,upper_r,lower_r,RBF_glo);
                    end
                    RF_evaluate = predict(RF_net,OffDec);

                    [id_selected,pred_rbf,feature_rbf,alignInfo] = SelectCandidateAGES( ...
                        OffDec,RBF_evaluate,RF_evaluate,Train_X,Train_Y,Problem, ...
                        upper_r,lower_r,TopK,MinAlign,Lambda,Beta,currentBest,AlignDB,'RBF');
                    feBefore = Problem.FE;
                    New      = Problem.Evaluation(OffDec(id_selected,:));
                    SurLog   = LogSurrogatePrediction(SurLog,'RBF',pred_rbf,New.objs,Problem.FE);
                    [AlignDB,AlignLog] = LogAlignmentEvaluation(AlignDB,AlignLog,feature_rbf,'RBF',pred_rbf,New.objs,Problem.FE,alignInfo);
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RBF',feBefore,currentBest);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];
                    RBF_evaluate(id_selected,:) = [];
                    RF_evaluate(id_selected,:)  = [];

                    [id_selected,pred_rf,feature_rf,alignInfo] = SelectCandidateAGES( ...
                        OffDec,RBF_evaluate,RF_evaluate,Train_X,Train_Y,Problem, ...
                        upper_r,lower_r,TopK,MinAlign,Lambda,Beta,currentBest,AlignDB,'RF');
                    feBefore = Problem.FE;
                    New      = Problem.Evaluation(OffDec(id_selected,:));
                    SurLog   = LogSurrogatePrediction(SurLog,'RF',pred_rf,New.objs,Problem.FE);
                    [AlignDB,AlignLog] = LogAlignmentEvaluation(AlignDB,AlignLog,feature_rf,'RF',pred_rf,New.objs,Problem.FE,alignInfo);
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RF',feBefore,currentBest);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];

                    id_selected = randi(size(OffDec,1));
                    feBefore    = Problem.FE;
                    New         = Problem.Evaluation(OffDec(id_selected,:));
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'Random',feBefore,currentBest);
                    DB = [DB, New];

                    [~,id_best] = min(DB.objs);
                    if length(id_best) > 1
                        id_best = id_best(1,:);
                    end
                    best_x = DB(id_best).decs;
                    best_r = best_x(:,1:Problem.Dr);
                    best_c = ceil(best_x(:,Problem.Dr+1:Problem.D));

                    Train_x    = DB.decs;
                    idx_local  = dist_ham(best_c,ceil(Train_x(:,Problem.Dr+1:Problem.D)));
                    best_Arc_r = Train_x(idx_local,1:Problem.Dr);
                    best_Arc_f = DB(idx_local).objs;

                    Local_flag = sum(idx_local) > 5*Problem.Dr;
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

                    [~,rank] = sort(DB.objs);
                    P.decs = DB(rank(1:Np)).decs;
                    P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                    P.objs = DB(rank(1:Np)).objs;
                end
            catch err
                FinalizeLogs(History,EvalLog,SurLog,AlignLog,runStart,Problem,historyFile);
                rethrow(err);
            end
            FinalizeLogs(History,EvalLog,SurLog,AlignLog,runStart,Problem,historyFile);
        end
    end
end

function [idSelected,predSelected,featureSelected,info] = SelectCandidateAGES( ...
    OffDec,RBF_evaluate,RF_evaluate,Train_X,Train_Y,Problem,upper_r,lower_r, ...
    TopK,MinAlign,Lambda,Beta,currentBest,AlignDB,selector)

    n = size(OffDec,1);
    [Features,FeatureInfo] = BuildAlignmentFeatures( ...
        OffDec,RBF_evaluate,RF_evaluate,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest,selector);

    if strcmp(selector,'RBF')
        selectorPred = RBF_evaluate(:);
    else
        selectorPred = RF_evaluate(:);
    end

    [fallbackId,predSelected] = SelectBestFinite(selectorPred);
    featureSelected = Features(fallbackId,:);
    info = DefaultAlignInfo();
    info.Quality = FeatureInfo.Quality(fallbackId);
    info.Novelty = FeatureInfo.Novelty(fallbackId);

    finite = isfinite(selectorPred);
    if ~HasAlignmentData(AlignDB,MinAlign) || ~any(finite)
        idSelected = fallbackId;
        return;
    end

    [predLogError,modelOK] = PredictAlignmentError(AlignDB,Features);
    if ~modelOK
        idSelected = fallbackId;
        return;
    end

    candidateIdx = Shortlist(selectorPred,TopK);
    if isempty(candidateIdx)
        idSelected = fallbackId;
        return;
    end

    errorPenalty = UnitNormalize(predLogError);
    score = FeatureInfo.Quality - Lambda*errorPenalty + Beta*FeatureInfo.Novelty;
    score(~finite) = -inf;

    [~,bestLocal] = max(score(candidateIdx));
    idSelected = candidateIdx(bestLocal);
    predSelected = selectorPred(idSelected);
    featureSelected = Features(idSelected,:);

    info.UsedAlignment = true;
    info.PredLogError  = predLogError(idSelected);
    info.PredAbsError  = exp(predLogError(idSelected)) - 1;
    info.Score         = score(idSelected);
    info.Quality       = FeatureInfo.Quality(idSelected);
    info.Novelty       = FeatureInfo.Novelty(idSelected);
end

function [Features,Info] = BuildAlignmentFeatures( ...
    OffDec,RBF_evaluate,RF_evaluate,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest,selector)

    n = size(OffDec,1);
    Dr = Problem.Dr;
    Dc = Problem.Dc;

    xrange = upper_r - lower_r;
    xrange(xrange < eps) = eps;
    x_r = (OffDec(:,1:Dr) - lower_r)./xrange;
    x_r = min(max(x_r,0),1);

    x_c = ceil(OffDec(:,Dr+1:Problem.D));
    len_c = NormalizeLenC(Problem.len_c,Dc);
    crange = max(1,len_c - 1);
    x_c_scaled = (x_c - 1)./crange;
    x_c_scaled = min(max(x_c_scaled,0),1);

    rbfPred = RBF_evaluate(:);
    rfPred  = RF_evaluate(:);
    predDiff = abs(rbfPred - rfPred);
    rbfRank = NormalizedRank(rbfPred);
    rfRank  = NormalizedRank(rfPred);

    if strcmp(selector,'RBF')
        selectorPred = rbfPred;
        selectorFlag = ones(n,1);
    else
        selectorPred = rfPred;
        selectorFlag = zeros(n,1);
    end

    predImprovement = currentBest - selectorPred;
    [nearestDist,avgKDist,sameCatCount] = CandidateDistances(OffDec,Train_X,Problem,upper_r,lower_r);
    novelty = UnitNormalize(nearestDist);
    quality = 1 - NormalizedRank(selectorPred);
    quality(~isfinite(quality)) = 0;

    bestNorm = NormalizeWithReference(currentBest,Train_Y);
    sameCatSeen = sameCatCount > 0;
    sameCatScaled = min(1,sameCatCount./max(1,size(Train_X,1)));

    Features = [ ...
        x_r, ...
        x_c_scaled, ...
        UnitNormalize(rbfPred), ...
        UnitNormalize(rfPred), ...
        UnitNormalize(predDiff), ...
        rbfRank, ...
        rfRank, ...
        UnitNormalize(nearestDist), ...
        UnitNormalize(avgKDist), ...
        double(sameCatSeen), ...
        sameCatScaled, ...
        repmat(bestNorm,n,1), ...
        UnitNormalize(predImprovement), ...
        selectorFlag];
    Features = ReplaceNonFinite(Features,0.5);

    Info.Quality = quality;
    Info.Novelty = novelty;
end

function [nearestDist,avgKDist,sameCatCount] = CandidateDistances(OffDec,Train_X,Problem,upper_r,lower_r)
    n  = size(OffDec,1);
    nt = size(Train_X,1);
    Dr = Problem.Dr;

    xrange = upper_r - lower_r;
    xrange(xrange < eps) = eps;
    candR = (OffDec(:,1:Dr) - lower_r)./xrange;
    trainR = (Train_X(:,1:Dr) - lower_r)./xrange;
    candC = ceil(OffDec(:,Dr+1:Problem.D));
    trainC = ceil(Train_X(:,Dr+1:Problem.D));

    nearestDist = zeros(n,1);
    avgKDist = zeros(n,1);
    sameCatCount = zeros(n,1);
    k = min(5,nt);

    for i = 1:n
        distR = mean(abs(trainR - candR(i,:)),2);
        distC = mean(trainC ~= candC(i,:),2);
        mixedDist = 0.5*distR + 0.5*distC;
        sortedDist = sort(mixedDist);
        nearestDist(i) = sortedDist(1);
        avgKDist(i) = mean(sortedDist(1:k));
        sameCatCount(i) = sum(all(trainC == candC(i,:),2));
    end
end

function len_c = NormalizeLenC(len_c,Dc)
    len_c = len_c(:)';
    if isempty(len_c)
        len_c = 5*ones(1,Dc);
    elseif numel(len_c) < Dc
        len_c = [len_c,repmat(len_c(end),1,Dc-numel(len_c))];
    elseif numel(len_c) > Dc
        len_c = len_c(1:Dc);
    end
end

function [predLogError,ok] = PredictAlignmentError(AlignDB,Features)
    ok = false;
    predLogError = nan(size(Features,1),1);

    y = AlignDB.LogAbsError(:);
    X = AlignDB.Features;
    valid = all(isfinite(X),2) & isfinite(y);
    X = X(valid,:);
    y = y(valid,:);

    if size(X,1) < 2
        return;
    end

    if max(y) - min(y) < eps
        predLogError(:) = mean(y);
        ok = true;
        return;
    end

    try
        model = fitrensemble(X,y,'Method','Bag','NumLearningCycles',30,'Learners','tree');
        predLogError = predict(model,Features);
        predLogError = max(predLogError,0);
        ok = all(isfinite(predLogError));
    catch
        predLogError(:) = mean(y);
        ok = true;
    end
end

function tf = HasAlignmentData(AlignDB,MinAlign)
    tf = size(AlignDB.Features,1) >= MinAlign && ...
         numel(unique(round(AlignDB.LogAbsError*1e8)/1e8)) >= 2;
end

function idx = Shortlist(pred,TopK)
    pred = pred(:);
    finite = isfinite(pred);
    if ~any(finite)
        idx = [];
        return;
    end
    rankValue = pred;
    rankValue(~finite) = inf;
    [~,rank] = sort(rankValue);
    idx = rank(1:min(TopK,sum(finite)));
end

function [idSelected,predSelected] = SelectBestFinite(pred)
    pred = pred(:);
    finite = isfinite(pred);
    if any(finite)
        rankValue = pred;
        rankValue(~finite) = inf;
        [~,rank] = sort(rankValue);
        idSelected = rank(1);
        predSelected = pred(idSelected);
    else
        idSelected = randi(numel(pred));
        predSelected = nan;
    end
end

function out = NormalizedRank(values)
    values = values(:);
    n = numel(values);
    finite = isfinite(values);
    rankValue = values;
    rankValue(~finite) = inf;
    [~,order] = sort(rankValue);
    rankPos = zeros(n,1);
    rankPos(order) = 1:n;
    out = (rankPos - 1)./max(1,n - 1);
    out(~finite) = 1;
end

function out = UnitNormalize(values)
    values = values(:);
    out = 0.5*ones(size(values));
    finite = isfinite(values);
    if ~any(finite)
        return;
    end
    lo = min(values(finite));
    hi = max(values(finite));
    if hi - lo < eps
        out(finite) = 0.5;
    else
        out(finite) = (values(finite) - lo)./(hi - lo);
    end
end

function out = NormalizeWithReference(value,reference)
    reference = reference(:);
    finite = isfinite(reference);
    if ~isfinite(value) || ~any(finite)
        out = 0.5;
        return;
    end
    lo = min(reference(finite));
    hi = max(reference(finite));
    if hi - lo < eps
        out = 0.5;
    else
        out = min(max((value - lo)./(hi - lo),0),1);
    end
end

function X = ReplaceNonFinite(X,value)
    X(~isfinite(X)) = value;
end

function info = DefaultAlignInfo()
    info.UsedAlignment = false;
    info.PredLogError  = nan;
    info.PredAbsError  = nan;
    info.Score         = nan;
    info.Quality       = nan;
    info.Novelty       = nan;
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

function [History,EvalLog,SurLog,AlignLog] = InitializeLogs()
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

    AlignLog.FE            = [];
    AlignLog.Source        = {};
    AlignLog.UsedAlignment = [];
    AlignLog.PredObj       = [];
    AlignLog.RealObj       = [];
    AlignLog.AbsError      = [];
    AlignLog.PredAbsError  = [];
    AlignLog.Score         = [];
    AlignLog.Quality       = [];
    AlignLog.Novelty       = [];
end

function AlignDB = InitializeAlignmentDB()
    AlignDB.Features = [];
    AlignDB.LogAbsError = [];
    AlignDB.Source = {};
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

function [AlignDB,AlignLog] = LogAlignmentEvaluation(AlignDB,AlignLog,feature,source,predicted,realValue,fe,info)
    realValue = realValue(1);
    predicted = predicted(1);
    absError = abs(predicted - realValue);

    AlignLog.FE(end+1,1)            = fe;
    AlignLog.Source{end+1,1}        = source;
    AlignLog.UsedAlignment(end+1,1) = info.UsedAlignment;
    AlignLog.PredObj(end+1,1)       = predicted;
    AlignLog.RealObj(end+1,1)       = realValue;
    AlignLog.AbsError(end+1,1)      = absError;
    AlignLog.PredAbsError(end+1,1)  = info.PredAbsError;
    AlignLog.Score(end+1,1)         = info.Score;
    AlignLog.Quality(end+1,1)       = info.Quality;
    AlignLog.Novelty(end+1,1)       = info.Novelty;

    if isfinite(absError) && all(isfinite(feature))
        AlignDB.Features(end+1,:) = feature;
        AlignDB.LogAbsError(end+1,1) = log(1 + absError);
        AlignDB.Source{end+1,1} = source;
    end
end

function RunSummary = FinalizeLogs(History,EvalLog,SurLog,AlignLog,runStart,Problem,historyFile)
    RunSummary = BuildRunSummary(History,EvalLog,SurLog,AlignLog,runStart,Problem);
    save(historyFile,'History','EvalLog','SurLog','AlignLog','RunSummary');
    PrintRunSummary(RunSummary,historyFile);
end

function RunSummary = BuildRunSummary(History,EvalLog,SurLog,AlignLog,runStart,Problem)
    sources = {'RBF','RF','Random','LocalSearch'};

    if isempty(EvalLog.Obj)
        bestObj = nan;
        bestFE  = nan;
    else
        [bestObj,idBest] = min(EvalLog.Obj);
        bestFE = EvalLog.FE(idBest);
    end

    RunSummary.Algorithm     = 'AGES';
    RunSummary.BestObjective = bestObj;
    RunSummary.BestFE        = bestFE;
    RunSummary.Runtime       = toc(runStart);
    RunSummary.TotalFE       = Problem.FE;

    for i = 1:length(sources)
        source = sources{i};
        RunSummary.([source,'Evaluations'])  = CountSource(EvalLog,source);
        RunSummary.([source,'Improvements']) = CountImprovement(EvalLog,source);
    end

    [RunSummary.RBF_MAE,RunSummary.RBF_RMSE] = SurrogateError(SurLog,'RBF');
    [RunSummary.RF_MAE, RunSummary.RF_RMSE]  = SurrogateError(SurLog,'RF');
    [RunSummary.AlignmentError_MAE,RunSummary.AlignmentError_RMSE] = AlignmentError(AlignLog);
    RunSummary.AlignmentSamples = numel(AlignLog.FE);
    RunSummary.AlignmentGuidedEvaluations = sum(logical(AlignLog.UsedAlignment));
    RunSummary.AlignmentFallbackEvaluations = RunSummary.AlignmentSamples - RunSummary.AlignmentGuidedEvaluations;
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
        err = SurLog.Pred(idx) - SurLog.Real(idx);
        err = err(isfinite(err));
        if isempty(err)
            mae = nan;
            rmse = nan;
        else
            mae = mean(abs(err));
            rmse = sqrt(mean(err.^2));
        end
    else
        mae = nan;
        rmse = nan;
    end
end

function [mae,rmse] = AlignmentError(AlignLog)
    idx = isfinite(AlignLog.PredAbsError) & isfinite(AlignLog.AbsError);
    if any(idx)
        err = AlignLog.PredAbsError(idx) - AlignLog.AbsError(idx);
        mae = mean(abs(err));
        rmse = sqrt(mean(err.^2));
    else
        mae = nan;
        rmse = nan;
    end
end

function PrintRunSummary(RunSummary,historyFile)
    fprintf('\n========== AGES Run Summary ==========\n');
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

    fprintf('Alignment samples: %d\n',RunSummary.AlignmentSamples);
    fprintf('Alignment-guided evaluations: %d\n',RunSummary.AlignmentGuidedEvaluations);
    fprintf('Alignment fallback evaluations: %d\n',RunSummary.AlignmentFallbackEvaluations);
    fprintf('Alignment MAE: %.12g, Alignment RMSE: %.12g\n\n',RunSummary.AlignmentError_MAE,RunSummary.AlignmentError_RMSE);

    fprintf('RBF MAE: %.12g, RBF RMSE: %.12g\n',RunSummary.RBF_MAE,RunSummary.RBF_RMSE);
    fprintf('RF MAE: %.12g, RF RMSE: %.12g\n',RunSummary.RF_MAE,RunSummary.RF_RMSE);
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('======================================\n\n');
end
