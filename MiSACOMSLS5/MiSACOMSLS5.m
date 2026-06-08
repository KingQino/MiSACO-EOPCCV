classdef MiSACOMSLS5 < ALGORITHM
% <single> <real>  <expensive>
		% Multisurrogate-Assisted Ant Colony Optimization with refined adaptive rescue
	% Np                   --- 60   --- Size of parent population
	% No                   --- 100  --- Size of offspring population
	% LocalStartCount      --- 6    --- Maximum number of elite local-search starting points
	% LocalElitePool       --- 30   --- Elite pool used to select local-search starts
	% LocalMinSampleFactor --- 5    --- Minimum same-category samples per continuous dimension
		% LocalNeighborCount   --- 30   --- Nearest same-category samples used by each local RBF
		% LocalNoveltyWeight   --- 0.01 --- Small novelty reward when ranking local candidates
		% LocalPredImproveTol  --- 0.005 --- Minimum relative predicted improvement for non-best starts
		% LocalSwitchMargin    --- 0.08 --- Extra score gain needed to switch away from best-only LS
		% LocalSwitchStartTol  --- 0.20 --- Allowed relative true-objective gap for non-best starts
		% LocalAdaptiveWindow  --- 12   --- Number of recent LS evaluations for adaptive rate
		% LocalGoodRate        --- 0.45 --- Recent LS improvement rate that keeps multi-start active
		% LocalBadRate         --- 0.15 --- Recent LS improvement rate that blocks mild multi-start
		% LocalStagFE          --- 30   --- FE since last improvement before multi-start probing
		% LocalHardStagFE      --- 60   --- FE since last improvement that overrides low-rate block
		% LocalRescueStartFrac --- 0.70 --- Fraction of max FE before late rescue can start
		% LocalRescueStagFE    --- 60   --- FE since last improvement before late rescue
		% LocalRescueBestTol   --- 0.10 --- Best objective threshold for late rescue
		% LocalRescueStartTol  --- 0.35 --- Relaxed start-objective gap in late rescue
		% LocalRescueSwitchMargin --- 0.05 --- Relaxed switch margin in late rescue
		% LocalRescuePredImproveTol --- 0.02 --- Predicted improvement required in late rescue
		% LocalRescueMaxFailures --- 3 --- Consecutive failed rescue LS before cooldown
		% LocalRescueCooldownFE --- 45 --- Rescue cooldown length after failures

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
	                    LocalNeighborCount,LocalNoveltyWeight,LocalPredImproveTol, ...
	                    LocalSwitchMargin,LocalSwitchStartTol,LocalAdaptiveWindow, ...
	                    LocalGoodRate,LocalBadRate,LocalStagFE,LocalHardStagFE, ...
	                    LocalRescueStartFrac,LocalRescueStagFE,LocalRescueBestTol, ...
	                    LocalRescueStartTol,LocalRescueSwitchMargin, ...
	                    LocalRescuePredImproveTol,LocalRescueMaxFailures,LocalRescueCooldownFE] = ...
	                    Algorithm.ParameterSet(60,100,6,30,5,30,0.01,0.005,0.08,0.20,12,0.45,0.15,30,60,0.70,60,0.10,0.35,0.05,0.02,3,45);
	                LocalStartCount = max(1,round(LocalStartCount));
	                LocalElitePool = max(LocalStartCount,round(LocalElitePool));
	                LocalMinSampleFactor = max(1,round(LocalMinSampleFactor));
	                LocalNeighborCount = max(Problem.Dr+2,round(LocalNeighborCount));
	                LocalNoveltyWeight = max(0,LocalNoveltyWeight);
	                LocalPredImproveTol = max(0,LocalPredImproveTol);
	                LocalSwitchMargin = max(0,LocalSwitchMargin);
	                LocalSwitchStartTol = max(0,LocalSwitchStartTol);
	                LocalAdaptiveWindow = max(3,round(LocalAdaptiveWindow));
	                LocalGoodRate = min(1,max(0,LocalGoodRate));
	                LocalBadRate = min(LocalGoodRate,max(0,LocalBadRate));
	                LocalStagFE = max(1,round(LocalStagFE));
	                LocalHardStagFE = max(LocalStagFE,round(LocalHardStagFE));
	                LocalRescueStartFrac = min(1,max(0,LocalRescueStartFrac));
	                LocalRescueStagFE = max(1,round(LocalRescueStagFE));
	                LocalRescueBestTol = max(0,LocalRescueBestTol);
	                LocalRescueStartTol = max(LocalSwitchStartTol,LocalRescueStartTol);
	                LocalRescueSwitchMargin = min(LocalSwitchMargin,max(0,LocalRescueSwitchMargin));
	                LocalRescuePredImproveTol = max(LocalPredImproveTol,LocalRescuePredImproveTol);
	                LocalRescueMaxFailures = max(1,round(LocalRescueMaxFailures));
	                LocalRescueCooldownFE = max(1,round(LocalRescueCooldownFE));
	                MSLocalStats = InitializeMSLocalStats();
	                LSAdaptive = InitializeLSAdaptiveState(LocalAdaptiveWindow);
	                PI      = UniformPoint(Np,Problem.D,'Latin');
                feBefore = Problem.FE;
                DB     = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Np,1).*PI+repmat(Problem.lower,Np,1));
                [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,DB,'Init',feBefore,currentBest);
	                lastImproveFE = LastImprovementFE(EvalLog);
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
	                    oldBest     = currentBest;
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RBF',feBefore,currentBest);
	                    if currentBest < oldBest
	                        lastImproveFE = EvalLog.FE(end);
	                    end
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
	                    oldBest     = currentBest;
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RF',feBefore,currentBest);
	                    if currentBest < oldBest
	                        lastImproveFE = EvalLog.FE(end);
	                    end
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];
                    
                    % Random Selection
                    id_selected = randi(size(OffDec,1));
                    feBefore    = Problem.FE;
                    New         = Problem.Evaluation(OffDec(id_selected,:));
	                    oldBest     = currentBest;
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'Random',feBefore,currentBest);
	                    if currentBest < oldBest
	                        lastImproveFE = EvalLog.FE(end);
	                    end
                    DB = [DB, New];
                    
	                    %% Adaptive Multi-start Surrogate-Assisted Local Search
	                    adaptiveInfo = SelectAdaptiveLocalMode( ...
	                        LSAdaptive,Problem,currentBest,lastImproveFE,LocalStartCount, ...
	                        LocalGoodRate,LocalBadRate,LocalStagFE,LocalHardStagFE, ...
	                        LocalRescueStartFrac,LocalRescueStagFE,LocalRescueBestTol);
	                    effectiveSwitchMargin = LocalSwitchMargin;
	                    effectiveStartTol = LocalSwitchStartTol;
	                    effectivePredImproveTol = LocalPredImproveTol;
	                    if adaptiveInfo.RescueMode
	                        effectiveSwitchMargin = LocalRescueSwitchMargin;
	                        effectiveStartTol = LocalRescueStartTol;
	                        effectivePredImproveTol = LocalRescuePredImproveTol;
	                    end
	                    [xLocal,local_success,localInfo] = MultiStartLocalSearch( ...
	                        DB,Problem,upper_r,lower_r,adaptiveInfo.StartCount,LocalElitePool, ...
	                        LocalMinSampleFactor,LocalNeighborCount,LocalNoveltyWeight, ...
	                        effectivePredImproveTol,effectiveSwitchMargin,effectiveStartTol, ...
	                        adaptiveInfo.AllowMultiStart);
	                    localInfo = AttachAdaptiveInfo(localInfo,adaptiveInfo);
	                    MSLocalStats = UpdateMSLocalStats(MSLocalStats,localInfo);
	                    if local_success
	                        feBefore = Problem.FE;
	                        New      = Problem.Evaluation(xLocal);
	                        oldBest  = currentBest;
	                        if localInfo.SelectedFromBestStart
	                            localSource = 'LocalSearchBestOnly';
	                        else
	                            localSource = 'LocalSearchMultiStart';
	                        end
	                        [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,localSource,feBefore,currentBest);
	                        lsImproved = currentBest < oldBest;
	                        LSAdaptive = UpdateLSAdaptiveState( ...
	                            LSAdaptive,lsImproved,adaptiveInfo,localInfo,Problem.FE, ...
	                            LocalRescueMaxFailures,LocalRescueCooldownFE);
	                        if lsImproved
	                            lastImproveFE = EvalLog.FE(end);
	                        end
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
	    DB,Problem,upper_r,lower_r,startCount,elitePool,minSampleFactor,neighborCount, ...
	    noveltyWeight,predImproveTol,switchMargin,switchStartTol,allowMultiStart)
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
	    predStartValues = [];
	    noveltyValues = [];
	    startObjectives = [];
	    isBestStart = [];
	    passGate = [];
	    attempted = 0;
	    [~,globalBestIdx] = min(Train_y);
	    currentBest = Train_y(globalBestIdx);
	
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
	
	        [pred,predStart] = PredictLocalCandidate(local_r,local_f,xrLocal,start_r);
	        if ~isfinite(pred) || ~isfinite(predStart)
	            continue;
	        end
	        predImprovement = predStart - pred;
	        improveMargin = predImproveTol*max(1,abs(predStart));

	        novelty = MixedNovelty(candidate,Train_x,Problem,upper_r,lower_r);
	        candidates = [candidates; candidate]; %#ok<AGROW>
	        predValues = [predValues; pred]; %#ok<AGROW>
	        predStartValues = [predStartValues; predStart]; %#ok<AGROW>
	        noveltyValues = [noveltyValues; novelty]; %#ok<AGROW>
	        startObjectives = [startObjectives; Train_y(startIdx(i))]; %#ok<AGROW>
	        isBestStart = [isBestStart; startIdx(i) == globalBestIdx]; %#ok<AGROW>
	        passGate = [passGate; startIdx(i) == globalBestIdx || predImprovement > improveMargin]; %#ok<AGROW>
	    end

	    info.Attempts = attempted;
	    info.Candidates = size(candidates,1);
	    info.BestCandidates = sum(isBestStart);
	    info.MultiCandidates = sum(~isBestStart);
	    info.GatedCandidates = sum(passGate);
	    if isempty(candidates)
	        return;
	    end
	
	    scale = std(Train_y(isfinite(Train_y)));
	    if ~isfinite(scale) || scale <= 0
	        scale = max(1,abs(median(Train_y(isfinite(Train_y)))));
	    end
	    score = predValues - noveltyWeight*scale*UnitNormalize(noveltyValues);
	    bestIds = find(isBestStart);
	    multiIds = find(~isBestStart & passGate);
	    if ~allowMultiStart
	        multiIds = [];
	    end
	    startSlack = switchStartTol*max(1,abs(currentBest));
	    multiIds = multiIds(startObjectives(multiIds) <= currentBest + startSlack);
	    if ~isempty(bestIds)
	        [~,bestLocal] = min(score(bestIds));
	        idBestOnly = bestIds(bestLocal);
	    else
	        idBestOnly = [];
	    end
	    if ~isempty(multiIds)
	        [~,bestLocal] = min(score(multiIds));
	        idBestMulti = multiIds(bestLocal);
	    else
	        idBestMulti = [];
	    end
	
	    if ~isempty(idBestOnly) && ~isempty(idBestMulti)
	        switchThreshold = switchMargin*max(1,abs(score(idBestOnly)));
	        if score(idBestMulti) < score(idBestOnly) - switchThreshold
	            idBest = idBestMulti;
	            info.Switches = 1;
	        else
	            idBest = idBestOnly;
	            info.Fallbacks = 1;
	        end
	    elseif ~isempty(idBestOnly)
	        idBest = idBestOnly;
	        info.Fallbacks = 1;
	    elseif ~isempty(idBestMulti)
	        idBest = idBestMulti;
	        info.Switches = 1;
	    else
	        info.SkippedByGate = 1;
	        return;
	    end
	
	    xSelected = candidates(idBest,:);
	    success = true;
	    info.SelectedPred = predValues(idBest);
	    info.SelectedPredStart = predStartValues(idBest);
	    info.SelectedPredImprovement = predStartValues(idBest) - predValues(idBest);
	    info.SelectedNovelty = noveltyValues(idBest);
	    info.SelectedStartObjective = startObjectives(idBest);
	    info.SelectedFromBestStart = isBestStart(idBest);
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
	
	function [pred,predStart] = PredictLocalCandidate(local_r,local_f,xrLocal,start_r)
	    pred = nan;
	    predStart = nan;
	    try
	        net = RBFCreate(local_r,local_f,'cubic');
	        pred = RBFInterp(xrLocal,net);
	        pred = pred(1);
	        predStart = RBFInterp(start_r,net);
	        predStart = predStart(1);
	    catch
	        pred = nan;
	        predStart = nan;
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
	    info.BestCandidates = 0;
	    info.MultiCandidates = 0;
	    info.GatedCandidates = 0;
	    info.Switches = 0;
	    info.Fallbacks = 0;
	    info.SkippedByGate = 0;
	    info.AdaptiveBestOnlyMode = 0;
	    info.AdaptiveMultiMode = 0;
	    info.AdaptiveStagnationTrigger = 0;
	    info.AdaptiveGoodRateTrigger = 0;
	    info.AdaptiveLowRateBlock = 0;
	    info.AdaptiveRescueMode = 0;
	    info.AdaptiveRescueCooldownMode = 0;
	    info.AdaptiveRecentRate = nan;
	    info.AdaptiveStagnationFE = nan;
	    info.SelectedPred = nan;
	    info.SelectedPredStart = nan;
	    info.SelectedPredImprovement = nan;
	    info.SelectedNovelty = nan;
	    info.SelectedStartObjective = nan;
	    info.SelectedFromBestStart = false;
	end
	
	function stats = InitializeMSLocalStats()
	    stats.Starts = 0;
	    stats.Attempts = 0;
	    stats.Candidates = 0;
	    stats.BestCandidates = 0;
	    stats.MultiCandidates = 0;
	    stats.GatedCandidates = 0;
	    stats.Successes = 0;
	    stats.Switches = 0;
	    stats.Fallbacks = 0;
	    stats.SkippedByGate = 0;
	    stats.AdaptiveBestOnlyModes = 0;
	    stats.AdaptiveMultiModes = 0;
	    stats.AdaptiveStagnationTriggers = 0;
	    stats.AdaptiveGoodRateTriggers = 0;
	    stats.AdaptiveLowRateBlocks = 0;
	    stats.AdaptiveRescueModes = 0;
	    stats.AdaptiveRescueCooldownModes = 0;
	    stats.AdaptiveRecentRates = [];
	    stats.AdaptiveStagnationFEs = [];
	    stats.SelectedPred = [];
	    stats.SelectedPredStart = [];
	    stats.SelectedPredImprovement = [];
	    stats.SelectedNovelty = [];
	    stats.SelectedStartObjective = [];
	    stats.SelectedFromBestStart = [];
	end
	
	function stats = UpdateMSLocalStats(stats,info)
	    stats.Starts = stats.Starts + info.Starts;
	    stats.Attempts = stats.Attempts + info.Attempts;
	    stats.Candidates = stats.Candidates + info.Candidates;
	    stats.BestCandidates = stats.BestCandidates + info.BestCandidates;
	    stats.MultiCandidates = stats.MultiCandidates + info.MultiCandidates;
	    stats.GatedCandidates = stats.GatedCandidates + info.GatedCandidates;
	    stats.Switches = stats.Switches + info.Switches;
	    stats.Fallbacks = stats.Fallbacks + info.Fallbacks;
	    stats.SkippedByGate = stats.SkippedByGate + info.SkippedByGate;
	    stats.AdaptiveBestOnlyModes = stats.AdaptiveBestOnlyModes + info.AdaptiveBestOnlyMode;
	    stats.AdaptiveMultiModes = stats.AdaptiveMultiModes + info.AdaptiveMultiMode;
	    stats.AdaptiveStagnationTriggers = stats.AdaptiveStagnationTriggers + info.AdaptiveStagnationTrigger;
	    stats.AdaptiveGoodRateTriggers = stats.AdaptiveGoodRateTriggers + info.AdaptiveGoodRateTrigger;
	    stats.AdaptiveLowRateBlocks = stats.AdaptiveLowRateBlocks + info.AdaptiveLowRateBlock;
	    stats.AdaptiveRescueModes = stats.AdaptiveRescueModes + info.AdaptiveRescueMode;
	    stats.AdaptiveRescueCooldownModes = stats.AdaptiveRescueCooldownModes + info.AdaptiveRescueCooldownMode;
	    if isfinite(info.AdaptiveRecentRate)
	        stats.AdaptiveRecentRates(end+1,1) = info.AdaptiveRecentRate;
	    end
	    if isfinite(info.AdaptiveStagnationFE)
	        stats.AdaptiveStagnationFEs(end+1,1) = info.AdaptiveStagnationFE;
	    end
	    if isfinite(info.SelectedPred)
	        stats.Successes = stats.Successes + 1;
	        stats.SelectedPred(end+1,1) = info.SelectedPred;
	        stats.SelectedPredStart(end+1,1) = info.SelectedPredStart;
	        stats.SelectedPredImprovement(end+1,1) = info.SelectedPredImprovement;
	        stats.SelectedNovelty(end+1,1) = info.SelectedNovelty;
	        stats.SelectedStartObjective(end+1,1) = info.SelectedStartObjective;
	        stats.SelectedFromBestStart(end+1,1) = info.SelectedFromBestStart;
	    end
	end

	function state = InitializeLSAdaptiveState(windowSize)
	    state.WindowSize = windowSize;
	    state.Outcomes = false(0,1);
	    state.RescueFailures = 0;
	    state.RescueCooldownUntilFE = -inf;
	end

	function state = UpdateLSAdaptiveState( ...
	    state,isImprovement,adaptiveInfo,localInfo,currentFE,rescueMaxFailures,rescueCooldownFE)
	    state.Outcomes(end+1,1) = logical(isImprovement);
	    if numel(state.Outcomes) > state.WindowSize
	        state.Outcomes = state.Outcomes(end-state.WindowSize+1:end);
	    end
	    if adaptiveInfo.RescueMode
	        if isImprovement
	            state.RescueFailures = 0;
	        elseif ~localInfo.SelectedFromBestStart
	            state.RescueFailures = state.RescueFailures + 1;
	            if state.RescueFailures >= rescueMaxFailures
	                state.RescueCooldownUntilFE = currentFE + rescueCooldownFE;
	                state.RescueFailures = 0;
	            end
	        end
	    end
	end

	function info = SelectAdaptiveLocalMode( ...
	    state,Problem,currentBest,lastImproveFE,maxStartCount,goodRate,badRate, ...
	    stagFE,hardStagFE,rescueStartFrac,rescueStagFE,rescueBestTol)
	    currentFE = Problem.FE;
	    maxFE = GetMaxFE(Problem,currentFE);
	    recentCount = numel(state.Outcomes);
	    if recentCount == 0
	        recentRate = nan;
	    else
	        recentRate = mean(state.Outcomes);
	    end
	    stagnationFE = max(0,currentFE - lastImproveFE);
	    enoughRecent = recentCount >= max(3,ceil(state.WindowSize/2));
	    isGoodRecent = enoughRecent && recentRate >= goodRate;
	    isLowRecent = enoughRecent && recentRate < badRate;
	    isStagnating = stagnationFE >= stagFE;
	    isHardStagnating = stagnationFE >= hardStagFE;
	    isLate = currentFE >= rescueStartFrac*maxFE;
	    rescueCooldown = currentFE < state.RescueCooldownUntilFE;
	    rescueMode = isLate && stagnationFE >= rescueStagFE && ...
	        currentBest > rescueBestTol && ~rescueCooldown;
	    lowRateBlock = isLowRecent && ~isHardStagnating && ~rescueMode;
	    allowMulti = (isGoodRecent || isStagnating || isHardStagnating || rescueMode) && ~lowRateBlock;
	
	    info.AllowMultiStart = allowMulti;
	    if allowMulti
	        info.StartCount = maxStartCount;
	    else
	        info.StartCount = 1;
	    end
	    info.RecentRate = recentRate;
	    info.RecentCount = recentCount;
	    info.StagnationFE = stagnationFE;
	    info.StagnationTrigger = isStagnating || isHardStagnating;
	    info.GoodRateTrigger = isGoodRecent;
	    info.LowRateBlock = lowRateBlock;
	    info.RescueMode = rescueMode;
	    info.RescueCooldownActive = rescueCooldown;
	end

	function localInfo = AttachAdaptiveInfo(localInfo,adaptiveInfo)
	    localInfo.AdaptiveBestOnlyMode = ~adaptiveInfo.AllowMultiStart;
	    localInfo.AdaptiveMultiMode = adaptiveInfo.AllowMultiStart;
	    localInfo.AdaptiveStagnationTrigger = adaptiveInfo.StagnationTrigger;
	    localInfo.AdaptiveGoodRateTrigger = adaptiveInfo.GoodRateTrigger;
	    localInfo.AdaptiveLowRateBlock = adaptiveInfo.LowRateBlock;
	    localInfo.AdaptiveRescueMode = adaptiveInfo.RescueMode;
	    localInfo.AdaptiveRescueCooldownMode = adaptiveInfo.RescueCooldownActive;
	    localInfo.AdaptiveRecentRate = adaptiveInfo.RecentRate;
	    localInfo.AdaptiveStagnationFE = adaptiveInfo.StagnationFE;
	end

	function maxFE = GetMaxFE(Problem,currentFE)
	    maxFE = max(currentFE,600);
	    try
	        maxFE = Problem.maxFE;
	    catch
	    end
	    if isempty(maxFE) || ~isfinite(maxFE) || maxFE <= 0
	        maxFE = max(currentFE,600);
	    end
	end

	function fe = LastImprovementFE(EvalLog)
	    if isempty(EvalLog.Obj)
	        fe = 0;
	        return;
	    end
	    bestHistory = cummin(EvalLog.Obj(:));
	    idx = find(EvalLog.Obj(:) == bestHistory & logical(EvalLog.Improved(:)),1,'last');
	    if isempty(idx)
	        [~,idx] = min(EvalLog.Obj(:));
	    end
	    fe = EvalLog.FE(idx);
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
	    RunSummary.MSLocalBestCandidates = MSLocalStats.BestCandidates;
	    RunSummary.MSLocalMultiCandidates = MSLocalStats.MultiCandidates;
	    RunSummary.MSLocalGatedCandidates = MSLocalStats.GatedCandidates;
	    RunSummary.MSLocalSuccesses = MSLocalStats.Successes;
	    RunSummary.MSLocalSwitches = MSLocalStats.Switches;
	    RunSummary.MSLocalFallbacks = MSLocalStats.Fallbacks;
	    RunSummary.MSLocalSkippedByGate = MSLocalStats.SkippedByGate;
	    RunSummary.MSLocalAdaptiveBestOnlyModes = MSLocalStats.AdaptiveBestOnlyModes;
	    RunSummary.MSLocalAdaptiveMultiModes = MSLocalStats.AdaptiveMultiModes;
	    RunSummary.MSLocalAdaptiveStagnationTriggers = MSLocalStats.AdaptiveStagnationTriggers;
	    RunSummary.MSLocalAdaptiveGoodRateTriggers = MSLocalStats.AdaptiveGoodRateTriggers;
	    RunSummary.MSLocalAdaptiveLowRateBlocks = MSLocalStats.AdaptiveLowRateBlocks;
	    RunSummary.MSLocalAdaptiveRescueModes = MSLocalStats.AdaptiveRescueModes;
	    RunSummary.MSLocalAdaptiveRescueCooldownModes = MSLocalStats.AdaptiveRescueCooldownModes;
	    RunSummary.MSLocalAdaptiveMeanRecentRate = mean(MSLocalStats.AdaptiveRecentRates,'omitnan');
	    RunSummary.MSLocalAdaptiveMeanStagnationFE = mean(MSLocalStats.AdaptiveStagnationFEs,'omitnan');
	    RunSummary.MSLocalMeanSelectedPred = mean(MSLocalStats.SelectedPred,'omitnan');
	    RunSummary.MSLocalMeanSelectedPredStart = mean(MSLocalStats.SelectedPredStart,'omitnan');
	    RunSummary.MSLocalMeanSelectedPredImprovement = mean(MSLocalStats.SelectedPredImprovement,'omitnan');
	    RunSummary.MSLocalMeanSelectedNovelty = mean(MSLocalStats.SelectedNovelty,'omitnan');
	    RunSummary.MSLocalMeanSelectedStartObjective = mean(MSLocalStats.SelectedStartObjective,'omitnan');
	    RunSummary.MSLocalBestStartSelectionRate = mean(MSLocalStats.SelectedFromBestStart,'omitnan');

	    sources = {'RBF','RF','Random','LocalSearchBestOnly','LocalSearchMultiStart'};
    for i = 1 : length(sources)
        source = sources{i};
        RunSummary.([source,'Evaluations'])  = CountSource(EvalLog,source);
        RunSummary.([source,'Improvements']) = CountImprovement(EvalLog,source);
    end
	    RunSummary.LocalSearchEvaluations = RunSummary.LocalSearchBestOnlyEvaluations + ...
	        RunSummary.LocalSearchMultiStartEvaluations;
	    RunSummary.LocalSearchImprovements = RunSummary.LocalSearchBestOnlyImprovements + ...
	        RunSummary.LocalSearchMultiStartImprovements;

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
	    fprintf('\n========== MiSACOMSLS5 Run Summary ==========\n');
    fprintf('Best Objective: %.12g\n',RunSummary.BestObjective);
    fprintf('Best FE: %d\n',RunSummary.BestFE);
    fprintf('Runtime: %.2fs\n',RunSummary.Runtime);
    fprintf('Total FE: %d\n\n',RunSummary.TotalFE);

	    fprintf('RBF Evaluations: %d\n',RunSummary.RBFEvaluations);
	    fprintf('RF Evaluations: %d\n',RunSummary.RFEvaluations);
	    fprintf('Random Evaluations: %d\n',RunSummary.RandomEvaluations);
	    fprintf('LocalSearch Evaluations: %d\n\n',RunSummary.LocalSearchEvaluations);
	    fprintf('LocalSearchBestOnly Evaluations: %d\n',RunSummary.LocalSearchBestOnlyEvaluations);
	    fprintf('LocalSearchMultiStart Evaluations: %d\n\n',RunSummary.LocalSearchMultiStartEvaluations);
	    fprintf('MS local start attempts: %d\n',RunSummary.MSLocalStartAttempts);
	    fprintf('MS local surrogate attempts: %d\n',RunSummary.MSLocalSurrogateAttempts);
	    fprintf('MS local candidates: %d\n',RunSummary.MSLocalCandidates);
	    fprintf('MS local gated candidates: %d\n',RunSummary.MSLocalGatedCandidates);
	    fprintf('MS local successes: %d\n\n',RunSummary.MSLocalSuccesses);
	    fprintf('MS local switches: %d\n',RunSummary.MSLocalSwitches);
	    fprintf('MS local best-only fallbacks: %d\n',RunSummary.MSLocalFallbacks);
	    fprintf('MS local skipped by gate: %d\n',RunSummary.MSLocalSkippedByGate);
	    fprintf('MS local adaptive best-only modes: %d\n',RunSummary.MSLocalAdaptiveBestOnlyModes);
	    fprintf('MS local adaptive multi modes: %d\n',RunSummary.MSLocalAdaptiveMultiModes);
	    fprintf('MS local adaptive stagnation triggers: %d\n',RunSummary.MSLocalAdaptiveStagnationTriggers);
	    fprintf('MS local adaptive good-rate triggers: %d\n',RunSummary.MSLocalAdaptiveGoodRateTriggers);
	    fprintf('MS local adaptive low-rate blocks: %d\n',RunSummary.MSLocalAdaptiveLowRateBlocks);
	    fprintf('MS local adaptive rescue modes: %d\n',RunSummary.MSLocalAdaptiveRescueModes);
	    fprintf('MS local adaptive rescue cooldown modes: %d\n',RunSummary.MSLocalAdaptiveRescueCooldownModes);
	    fprintf('MS local adaptive mean recent rate: %.4g\n',RunSummary.MSLocalAdaptiveMeanRecentRate);
	    fprintf('MS local adaptive mean stagnation FE: %.4g\n',RunSummary.MSLocalAdaptiveMeanStagnationFE);
	    fprintf('MS local best-start selection rate: %.4g\n\n',RunSummary.MSLocalBestStartSelectionRate);

	    fprintf('RBF Improvements: %d\n',RunSummary.RBFImprovements);
    fprintf('RF Improvements: %d\n',RunSummary.RFImprovements);
    fprintf('Random Improvements: %d\n',RunSummary.RandomImprovements);
    fprintf('LocalSearch Improvements: %d\n\n',RunSummary.LocalSearchImprovements);
    fprintf('LocalSearchBestOnly Improvements: %d\n',RunSummary.LocalSearchBestOnlyImprovements);
    fprintf('LocalSearchMultiStart Improvements: %d\n\n',RunSummary.LocalSearchMultiStartImprovements);

    fprintf('RBF MAE: %.12g, RBF RMSE: %.12g\n',RunSummary.RBF_MAE,RunSummary.RBF_RMSE);
    fprintf('RF MAE: %.12g, RF RMSE: %.12g\n',RunSummary.RF_MAE,RunSummary.RF_RMSE);
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('========================================\n\n');
end
