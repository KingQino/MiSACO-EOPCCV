classdef MiSACOMSLS < ALGORITHM
% <single> <real>  <expensive>
	% Multisurrogate-Assisted Ant Colony Optimization with multi-start local search
	% Np                   --- 60   --- Size of parent population
	% No                   --- 100  --- Size of offspring population
	% LocalStartCount      --- 6    --- Number of elite local-search starting points
	% LocalElitePool       --- 30   --- Elite pool used to select local-search starts
	% LocalMinSampleFactor --- 5    --- Minimum same-category samples per continuous dimension
	% LocalNeighborCount   --- 30   --- Nearest same-category samples used by each local RBF
	% LocalNoveltyWeight   --- 0.05 --- Small novelty reward when ranking local candidates

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
	                [Np,No,LocalStartCount,LocalElitePool,LocalMinSampleFactor, ...
	                    LocalNeighborCount,LocalNoveltyWeight] = ...
	                    Algorithm.ParameterSet(60,100,6,30,5,30,0.05);
	                LocalStartCount = max(1,round(LocalStartCount));
	                LocalElitePool = max(LocalStartCount,round(LocalElitePool));
	                LocalMinSampleFactor = max(1,round(LocalMinSampleFactor));
	                LocalNeighborCount = max(Problem.Dr+2,round(LocalNeighborCount));
	                LocalNoveltyWeight = max(0,LocalNoveltyWeight);
	                MSLocalStats = InitializeMSLocalStats();
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
                    
	                    %% Multi-start Surrogate-Assisted Local Search
	                    [xLocal,local_success,localInfo] = MultiStartLocalSearch( ...
	                        DB,Problem,upper_r,lower_r,LocalStartCount,LocalElitePool, ...
	                        LocalMinSampleFactor,LocalNeighborCount,LocalNoveltyWeight);
	                    MSLocalStats = UpdateMSLocalStats(MSLocalStats,localInfo);
	                    if local_success
	                        feBefore = Problem.FE;
	                        New      = Problem.Evaluation(xLocal);
	                        [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'LocalSearch',feBefore,currentBest);
	                        DB = [DB, New];
	                    end
                    
                    %% Population Update
                    [~,rank] = sort(DB.objs);
                    P.decs = DB(rank(1:Np)).decs;
                    P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                    P.objs = DB(rank(1:Np)).objs;
                end
            catch err
	                FinalizeLogs(History,EvalLog,SurLog,MSLocalStats,runStart,Problem,historyFile);
	                rethrow(err);
	            end
	            FinalizeLogs(History,EvalLog,SurLog,MSLocalStats,runStart,Problem,historyFile);
	        end
		end
	end

	function [xSelected,success,info] = MultiStartLocalSearch( ...
	    DB,Problem,upper_r,lower_r,startCount,elitePool,minSampleFactor,neighborCount,noveltyWeight)
	    success = false;
	    xSelected = [];
	    info = DefaultMSLocalInfo();
	
	    Train_x = DB.decs;
	    Train_x(:,Problem.Dr+1:Problem.D) = ceil(Train_x(:,Problem.Dr+1:Problem.D));
	    Train_y = DB.objs;
	    Train_y = Train_y(:,1);
	    startIdx = SelectLocalStartIndices(Train_x,Train_y,Problem,startCount,elitePool);
	    info.Starts = numel(startIdx);
	    if isempty(startIdx)
	        return;
	    end
	
	    minSamples = max(Problem.Dr + 2,minSampleFactor*Problem.Dr);
	    candidates = [];
	    predValues = [];
	    noveltyValues = [];
	    startObjectives = [];
	    attempted = 0;
	
	    for i = 1:numel(startIdx)
	        start_x = Train_x(startIdx(i),:);
	        start_r = start_x(:,1:Problem.Dr);
	        start_c = ceil(start_x(:,Problem.Dr+1:Problem.D));
	        idxLocal = SameCategory(start_c,ceil(Train_x(:,Problem.Dr+1:Problem.D)));
	        if sum(idxLocal) <= minSamples
	            continue;
	        end
	
	        arc_r = Train_x(idxLocal,1:Problem.Dr);
	        arc_f = Train_y(idxLocal,:);
	        idxNear = NearestContinuousRows(start_r,arc_r,neighborCount);
	        if sum(idxNear) < Problem.Dr + 2
	            continue;
	        end
	
	        local_r = arc_r(idxNear,:);
	        local_f = arc_f(idxNear,:);
	        local_up = max(local_r);
	        local_dn = min(local_r);
	        attempted = attempted + 1;
	        [xrLocal,localSuccess] = Local_Search(local_r,local_f,local_up,local_dn,start_r);
	        if ~localSuccess
	            continue;
	        end
	
	        candidate = [xrLocal,start_c];
	        candidate(:,Problem.Dr+1:Problem.D) = ceil(candidate(:,Problem.Dr+1:Problem.D));
	        if IsExistingDecision(candidate,Train_x,Problem,upper_r,lower_r)
	            continue;
	        end
	
	        pred = PredictLocalCandidate(local_r,local_f,xrLocal);
	        if ~isfinite(pred)
	            continue;
	        end
	
	        novelty = MixedNovelty(candidate,Train_x,Problem,upper_r,lower_r);
	        candidates = [candidates; candidate]; %#ok<AGROW>
	        predValues = [predValues; pred]; %#ok<AGROW>
	        noveltyValues = [noveltyValues; novelty]; %#ok<AGROW>
	        startObjectives = [startObjectives; Train_y(startIdx(i))]; %#ok<AGROW>
	    end
	
	    info.Attempts = attempted;
	    info.Candidates = size(candidates,1);
	    if isempty(candidates)
	        return;
	    end
	
	    scale = std(Train_y(isfinite(Train_y)));
	    if ~isfinite(scale) || scale <= 0
	        scale = max(1,abs(median(Train_y(isfinite(Train_y)))));
	    end
	    score = predValues - noveltyWeight*scale*UnitNormalize(noveltyValues);
	    [~,idBest] = min(score);
	    xSelected = candidates(idBest,:);
	    success = true;
	    info.SelectedPred = predValues(idBest);
	    info.SelectedNovelty = noveltyValues(idBest);
	    info.SelectedStartObjective = startObjectives(idBest);
	end
	
	function startIdx = SelectLocalStartIndices(Train_x,Train_y,Problem,startCount,elitePool)
	    finite = isfinite(Train_y);
	    if ~any(finite)
	        startIdx = [];
	        return;
	    end
	
	    validIdx = find(finite);
	    [~,rank] = sort(Train_y(validIdx),'ascend');
	    rankedIdx = validIdx(rank);
	    eliteIdx = rankedIdx(1:min(elitePool,numel(rankedIdx)));
	    startIdx = [];
	
	    catValues = ceil(Train_x(eliteIdx,Problem.Dr+1:Problem.D));
	    if isempty(catValues)
	        uniqueCatRows = 1;
	    else
	        [~,uniqueCatRows] = unique(catValues,'rows','stable');
	    end
	    startIdx = eliteIdx(uniqueCatRows(:))';
	    startIdx = startIdx(1:min(startCount,numel(startIdx)));
	
	    if numel(startIdx) < startCount
	        for i = 1:numel(eliteIdx)
	            if ~ismember(eliteIdx(i),startIdx)
	                startIdx(end+1) = eliteIdx(i); %#ok<AGROW>
	            end
	            if numel(startIdx) >= startCount
	                break;
	            end
	        end
	    end
	end
	
	function idx = SameCategory(xc,C)
	    if isempty(xc)
	        idx = true(size(C,1),1);
	    else
	        idx = all(C == xc,2);
	    end
	end
	
	function idx = NearestContinuousRows(x0,x,neighborCount)
	    if isempty(x)
	        idx = false(0,1);
	        return;
	    end
	    dist = sqrt(sum((x - x0).^2,2));
	    [~,sortRank] = sort(dist,'ascend');
	    idx = false(size(x,1),1);
	    idx(sortRank(1:min(neighborCount,numel(sortRank)))) = true;
	end
	
	function pred = PredictLocalCandidate(local_r,local_f,xrLocal)
	    pred = nan;
	    try
	        net = RBFCreate(local_r,local_f,'cubic');
	        pred = RBFInterp(xrLocal,net);
	        pred = pred(1);
	    catch
	        pred = nan;
	    end
	end
	
	function tf = IsExistingDecision(candidate,Train_x,Problem,upper_r,lower_r)
	    if isempty(Train_x)
	        tf = false;
	        return;
	    end
	    Dr = Problem.Dr;
	    xrange = upper_r - lower_r;
	    xrange(xrange < eps) = 1;
	    rDist = max(abs((Train_x(:,1:Dr) - candidate(:,1:Dr))./xrange),[],2);
	    sameC = SameCategory(ceil(candidate(:,Dr+1:Problem.D)),ceil(Train_x(:,Dr+1:Problem.D)));
	    tf = any(rDist <= 1e-10 & sameC);
	end
	
	function novelty = MixedNovelty(candidate,Train_x,Problem,upper_r,lower_r)
	    if isempty(Train_x)
	        novelty = 1;
	        return;
	    end
	    Dr = Problem.Dr;
	    Dc = Problem.Dc;
	    xrange = upper_r - lower_r;
	    xrange(xrange < eps) = 1;
	    rDist = sqrt(sum(((Train_x(:,1:Dr) - candidate(:,1:Dr))./xrange).^2,2))./sqrt(max(1,Dr));
	    if Dc > 0
	        cDist = mean(ceil(Train_x(:,Dr+1:Problem.D)) ~= ceil(candidate(:,Dr+1:Problem.D)),2);
	    else
	        cDist = zeros(size(Train_x,1),1);
	    end
	    catWeight = Dc./max(1,Problem.D);
	    mixedDist = (1 - catWeight).*rDist + catWeight.*cDist;
	    novelty = min(mixedDist);
	end
	
	function out = UnitNormalize(values)
	    values = values(:);
	    out = zeros(size(values));
	    finite = isfinite(values);
	    if ~any(finite)
	        return;
	    end
	    low = min(values(finite));
	    high = max(values(finite));
	    if high <= low
	        out(finite) = 0.5;
	    else
	        out(finite) = (values(finite) - low)./(high - low);
	    end
	end
	
	function info = DefaultMSLocalInfo()
	    info.Starts = 0;
	    info.Attempts = 0;
	    info.Candidates = 0;
	    info.SelectedPred = nan;
	    info.SelectedNovelty = nan;
	    info.SelectedStartObjective = nan;
	end
	
	function stats = InitializeMSLocalStats()
	    stats.Starts = 0;
	    stats.Attempts = 0;
	    stats.Candidates = 0;
	    stats.Successes = 0;
	    stats.SelectedPred = [];
	    stats.SelectedNovelty = [];
	    stats.SelectedStartObjective = [];
	end
	
	function stats = UpdateMSLocalStats(stats,info)
	    stats.Starts = stats.Starts + info.Starts;
	    stats.Attempts = stats.Attempts + info.Attempts;
	    stats.Candidates = stats.Candidates + info.Candidates;
	    if isfinite(info.SelectedPred)
	        stats.Successes = stats.Successes + 1;
	        stats.SelectedPred(end+1,1) = info.SelectedPred;
	        stats.SelectedNovelty(end+1,1) = info.SelectedNovelty;
	        stats.SelectedStartObjective(end+1,1) = info.SelectedStartObjective;
	    end
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

	function RunSummary = FinalizeLogs(History,EvalLog,SurLog,MSLocalStats,runStart,Problem,historyFile)
	    RunSummary = BuildRunSummary(History,EvalLog,SurLog,MSLocalStats,runStart,Problem);
	    save(historyFile,'History','EvalLog','SurLog','RunSummary');
	    PrintRunSummary(RunSummary,historyFile);
	end

	function RunSummary = BuildRunSummary(~,EvalLog,SurLog,MSLocalStats,runStart,Problem)
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
	    RunSummary.MSLocalStartAttempts = MSLocalStats.Starts;
	    RunSummary.MSLocalSurrogateAttempts = MSLocalStats.Attempts;
	    RunSummary.MSLocalCandidates = MSLocalStats.Candidates;
	    RunSummary.MSLocalSuccesses = MSLocalStats.Successes;
	    RunSummary.MSLocalMeanSelectedPred = mean(MSLocalStats.SelectedPred,'omitnan');
	    RunSummary.MSLocalMeanSelectedNovelty = mean(MSLocalStats.SelectedNovelty,'omitnan');
	    RunSummary.MSLocalMeanSelectedStartObjective = mean(MSLocalStats.SelectedStartObjective,'omitnan');

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
	    fprintf('\n========== MiSACOMSLS Run Summary ==========\n');
    fprintf('Best Objective: %.12g\n',RunSummary.BestObjective);
    fprintf('Best FE: %d\n',RunSummary.BestFE);
    fprintf('Runtime: %.2fs\n',RunSummary.Runtime);
    fprintf('Total FE: %d\n\n',RunSummary.TotalFE);

	    fprintf('RBF Evaluations: %d\n',RunSummary.RBFEvaluations);
	    fprintf('RF Evaluations: %d\n',RunSummary.RFEvaluations);
	    fprintf('Random Evaluations: %d\n',RunSummary.RandomEvaluations);
	    fprintf('LocalSearch Evaluations: %d\n\n',RunSummary.LocalSearchEvaluations);
	    fprintf('MS local start attempts: %d\n',RunSummary.MSLocalStartAttempts);
	    fprintf('MS local surrogate attempts: %d\n',RunSummary.MSLocalSurrogateAttempts);
	    fprintf('MS local candidates: %d\n',RunSummary.MSLocalCandidates);
	    fprintf('MS local successes: %d\n\n',RunSummary.MSLocalSuccesses);

	    fprintf('RBF Improvements: %d\n',RunSummary.RBFImprovements);
    fprintf('RF Improvements: %d\n',RunSummary.RFImprovements);
    fprintf('Random Improvements: %d\n',RunSummary.RandomImprovements);
    fprintf('LocalSearch Improvements: %d\n\n',RunSummary.LocalSearchImprovements);

    fprintf('RBF MAE: %.12g, RBF RMSE: %.12g\n',RunSummary.RBF_MAE,RunSummary.RBF_RMSE);
    fprintf('RF MAE: %.12g, RF RMSE: %.12g\n',RunSummary.RF_MAE,RunSummary.RF_RMSE);
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('========================================\n\n');
end
