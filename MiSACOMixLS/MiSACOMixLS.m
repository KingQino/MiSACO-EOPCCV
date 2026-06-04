classdef MiSACOMixLS < ALGORITHM
% <single> <real>  <expensive>
	% Multisurrogate-Assisted Ant Colony Optimization with conservative mixed local search
	% Np                   --- 60   --- Size of parent population
	% No                   --- 100  --- Size of offspring population
	% MixNeighborBudget    --- 24   --- Maximum discrete local neighbors
	% MixMaxHamming        --- 1    --- Maximum categorical variables changed
	% MixPredImproveTol    --- 0.01 --- Minimum relative predicted improvement for mixed challenger
	% MixSwitchMargin      --- 0.05 --- Extra predicted gain needed over continuous local search
	% MixNoveltyWeight     --- 0.02 --- Small mixed-novelty reward in challenger ranking

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
            MixLSStats = InitializeMixLSStats();

            try
                %% Generate random population
	                [Np,No,MixNeighborBudget,MixMaxHamming,MixPredImproveTol, ...
	                    MixSwitchMargin,MixNoveltyWeight] = ...
	                    Algorithm.ParameterSet(60,100,24,1,0.01,0.05,0.02);
	                MixNeighborBudget = max(0,round(MixNeighborBudget));
	                MixMaxHamming = max(1,round(MixMaxHamming));
	                MixPredImproveTol = max(0,MixPredImproveTol);
	                MixSwitchMargin = max(0,MixSwitchMargin);
	                MixNoveltyWeight = max(0,MixNoveltyWeight);
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
                    
	                    %% Conservative Mixed Local Search
	                    [xLocal,local_success,mixInfo] = MixedLocalSearch( ...
	                        DB,Problem,upper_r,lower_r,RBF_glo,RF_net, ...
	                        MixNeighborBudget,MixMaxHamming,MixPredImproveTol, ...
	                        MixSwitchMargin,MixNoveltyWeight);
	                    MixLSStats = UpdateMixLSStats(MixLSStats,mixInfo);
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
	                FinalizeLogs(History,EvalLog,SurLog,MixLSStats,runStart,Problem,historyFile);
	                rethrow(err);
	            end
	            FinalizeLogs(History,EvalLog,SurLog,MixLSStats,runStart,Problem,historyFile);
	        end
		end
	end

	function [xSelected,success,info] = MixedLocalSearch( ...
	    DB,Problem,upper_r,lower_r,RBF_glo,RF_net,neighborBudget,maxHamming, ...
	    predImproveTol,switchMargin,noveltyWeight)
	    success = false;
	    xSelected = [];
	    info = DefaultMixLSInfo();
	    info.Attempts = 1;
	
	    Train_x = DB.decs;
	    Train_x(:,Problem.Dr+1:Problem.D) = ceil(Train_x(:,Problem.Dr+1:Problem.D));
	    Train_y = DB.objs;
	    Train_y = Train_y(:,1);
	    [currentBest,idBest] = min(Train_y);
	    best_x = Train_x(idBest,:);
	    best_r = best_x(:,1:Problem.Dr);
	    best_c = ceil(best_x(:,Problem.Dr+1:Problem.D));
	
	    [baselineCandidate,baselineSuccess] = BuildContinuousLocalCandidate( ...
	        Train_x,Train_y,best_r,best_c,Problem);
	    if baselineSuccess
	        candidates = baselineCandidate;
	        labels = {'Continuous'};
	        info.BaselineCandidates = 1;
	    else
	        candidates = zeros(0,Problem.D);
	        labels = cell(0,1);
	    end
	
	    if neighborBudget > 0 && Problem.Dc > 0
	        eliteC = EliteCategories(Train_x,Train_y,Problem,30);
	        neighborC = GenerateDiscreteNeighbors( ...
	            best_c,eliteC,NormalizeLenC(Problem.len_c,Problem.Dc),neighborBudget,maxHamming);
	        rForMixed = best_r;
	        if baselineSuccess
	            rForMixed = baselineCandidate(:,1:Problem.Dr);
	        end
	        mixedCandidates = [repmat(rForMixed,size(neighborC,1),1),neighborC];
	        mixedCandidates = RemoveExistingAndDuplicate(mixedCandidates,Train_x,Problem,upper_r,lower_r);
	        info.DiscreteCandidates = size(mixedCandidates,1);
	        if ~isempty(mixedCandidates)
	            candidates = [candidates; mixedCandidates];
	            labels = [labels; repmat({'Mixed'},size(mixedCandidates,1),1)]; %#ok<AGROW>
	        end
	    end
	
	    info.TotalCandidates = size(candidates,1);
	    if isempty(candidates)
	        info.Skipped = 1;
	        return;
	    end
	
	    [predValues,noveltyValues] = ScoreMixedCandidates( ...
	        candidates,Train_x,Train_y,Problem,upper_r,lower_r,RBF_glo,RF_net,noveltyWeight);
	    finite = isfinite(predValues);
	    if ~any(finite)
	        info.Skipped = 1;
	        return;
	    end
	
	    score = predValues;
	    if noveltyWeight > 0
	        scale = std(Train_y(isfinite(Train_y)));
	        if ~isfinite(scale) || scale <= 0
	            scale = max(1,abs(median(Train_y(isfinite(Train_y)))));
	        end
	        score = score - noveltyWeight*scale*UnitNormalize(noveltyValues);
	    end
	    score(~finite) = inf;
	
	    baselineIdx = find(strcmp(labels,'Continuous'),1);
	    mixedIdx = find(strcmp(labels,'Mixed') & finite);
	    selectedIdx = [];
	    if ~isempty(baselineIdx) && isfinite(score(baselineIdx))
	        selectedIdx = baselineIdx;
	        mixedAllowed = false(size(mixedIdx));
	        if ~isempty(mixedIdx)
	            requiredPred = min(predValues(baselineIdx),currentBest) - ...
	                predImproveTol*max(1,abs(currentBest));
	            mixedAllowed = predValues(mixedIdx) < requiredPred;
	        end
	        allowedIdx = mixedIdx(mixedAllowed);
	        if ~isempty(allowedIdx)
	            [~,bestLocal] = min(score(allowedIdx));
	            challengerIdx = allowedIdx(bestLocal);
	            switchThreshold = switchMargin*max(1,abs(score(baselineIdx)));
	            if score(challengerIdx) < score(baselineIdx) - switchThreshold
	                selectedIdx = challengerIdx;
	            end
	        end
	    else
	        requiredPred = currentBest - predImproveTol*max(1,abs(currentBest));
	        allowedIdx = mixedIdx(predValues(mixedIdx) < requiredPred);
	        if ~isempty(allowedIdx)
	            [~,bestLocal] = min(score(allowedIdx));
	            selectedIdx = allowedIdx(bestLocal);
	        end
	    end
	
	    if isempty(selectedIdx)
	        info.Skipped = 1;
	        return;
	    end
	
	    xSelected = candidates(selectedIdx,:);
	    success = true;
	    info.SelectedPred = predValues(selectedIdx);
	    info.SelectedNovelty = noveltyValues(selectedIdx);
	    info.SelectedIsMixed = strcmp(labels{selectedIdx},'Mixed');
	    if info.SelectedIsMixed
	        info.SelectedMixed = 1;
	    else
	        info.SelectedBaseline = 1;
	    end
	end
	
	function [candidate,success] = BuildContinuousLocalCandidate(Train_x,Train_y,best_r,best_c,Problem)
	    candidate = [];
	    success = false;
	    idxLocal = SameCategory(best_c,ceil(Train_x(:,Problem.Dr+1:Problem.D)));
	    minSamples = 5*Problem.Dr;
	    if sum(idxLocal) <= minSamples
	        return;
	    end
	    arc_r = Train_x(idxLocal,1:Problem.Dr);
	    arc_f = Train_y(idxLocal,:);
	    idxNear = NearestContinuousRows(best_r,arc_r,30);
	    local_r = arc_r(idxNear,:);
	    local_f = arc_f(idxNear,:);
	    if size(local_r,1) < Problem.Dr + 2
	        return;
	    end
	    local_up = max(local_r);
	    local_dn = min(local_r);
	    [xrLocal,localSuccess] = Local_Search(local_r,local_f,local_up,local_dn,best_r);
	    if localSuccess
	        candidate = [xrLocal,best_c];
	        success = true;
	    end
	end
	
	function eliteC = EliteCategories(Train_x,Train_y,Problem,elitePool)
	    [~,rank] = sort(Train_y,'ascend');
	    eliteIdx = rank(1:min(elitePool,numel(rank)));
	    eliteC = ceil(Train_x(eliteIdx,Problem.Dr+1:Problem.D));
	    if ~isempty(eliteC)
	        eliteC = unique(eliteC,'rows','stable');
	    end
	end
	
	function neighborC = GenerateDiscreteNeighbors(best_c,eliteC,lenC,budget,maxHamming)
	    Dc = numel(best_c);
	    neighborC = zeros(0,Dc);
	    for j = 1:Dc
	        for value = 1:lenC(j)
	            if value == best_c(j)
	                continue;
	            end
	            c = best_c;
	            c(j) = value;
	            neighborC = [neighborC; c]; %#ok<AGROW>
	        end
	    end
	    if maxHamming >= 2 && Dc >= 2
	        for j = 1:Dc-1
	            for k = j+1:Dc
	                for valueJ = 1:lenC(j)
	                    for valueK = 1:lenC(k)
	                        if valueJ == best_c(j) && valueK == best_c(k)
	                            continue;
	                        end
	                        c = best_c;
	                        c(j) = valueJ;
	                        c(k) = valueK;
	                        if sum(c ~= best_c) <= maxHamming
	                            neighborC = [neighborC; c]; %#ok<AGROW>
	                        end
	                    end
	                end
	            end
	        end
	    end
	    if ~isempty(eliteC)
	        distElite = sum(eliteC ~= best_c,2);
	        eliteC = eliteC(distElite > 0 & distElite <= maxHamming,:);
	        neighborC = [neighborC; eliteC];
	    end
	    if isempty(neighborC)
	        return;
	    end
	    neighborC = unique(neighborC,'rows','stable');
	    if size(neighborC,1) > budget
	        neighborC = neighborC(1:budget,:);
	    end
	end
	
	function candidates = RemoveExistingAndDuplicate(candidates,Train_x,Problem,upper_r,lower_r)
	    if isempty(candidates)
	        return;
	    end
	    candidates(:,Problem.Dr+1:Problem.D) = ceil(candidates(:,Problem.Dr+1:Problem.D));
	    candidates = unique(candidates,'rows','stable');
	    keep = true(size(candidates,1),1);
	    for i = 1:size(candidates,1)
	        keep(i) = ~IsExistingDecision(candidates(i,:),Train_x,Problem,upper_r,lower_r);
	    end
	    candidates = candidates(keep,:);
	end
	
	function [predValues,noveltyValues] = ScoreMixedCandidates( ...
	    candidates,Train_x,Train_y,Problem,upper_r,lower_r,RBF_glo,RF_net,~)
	    n = size(candidates,1);
	    predRBF = nan(n,1);
	    predRF = nan(n,1);
	    rbfModel = RBF_glo;
	    try
	        rbfModel = RBFCreate_Hamming(Train_x,Train_y,Problem.Dc,upper_r,lower_r,'gaussian');
	    catch
	    end
	    for i = 1:n
	        try
	            predRBF(i) = RBFInterp_Hamming(candidates(i,:),Problem.Dc,upper_r,lower_r,rbfModel);
	        catch
	            predRBF(i) = nan;
	        end
	    end
	    try
	        predRF = predict(RF_net,candidates);
	    catch
	        predRF(:) = nan;
	    end
	    predValues = RowMedianFinite([predRBF,predRF]);
	    noveltyValues = zeros(n,1);
	    for i = 1:n
	        noveltyValues(i) = MixedNovelty(candidates(i,:),Train_x,Problem,upper_r,lower_r);
	    end
	end
	
	function out = RowMedianFinite(values)
	    out = nan(size(values,1),1);
	    for i = 1:size(values,1)
	        row = values(i,:);
	        row = row(isfinite(row));
	        if ~isempty(row)
	            out(i) = median(row);
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
	
	function lenC = NormalizeLenC(len_c,Dc)
	    if Dc <= 0
	        lenC = zeros(1,0);
	        return;
	    end
	    if isempty(len_c)
	        lenC = ones(1,Dc);
	    elseif isscalar(len_c)
	        lenC = repmat(len_c,1,Dc);
	    else
	        lenC = len_c(:)';
	        if numel(lenC) < Dc
	            lenC = [lenC,repmat(lenC(end),1,Dc-numel(lenC))];
	        else
	            lenC = lenC(1:Dc);
	        end
	    end
	    lenC = max(1,round(lenC));
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
	
	function info = DefaultMixLSInfo()
	    info.Attempts = 0;
	    info.BaselineCandidates = 0;
	    info.DiscreteCandidates = 0;
	    info.TotalCandidates = 0;
	    info.SelectedBaseline = 0;
	    info.SelectedMixed = 0;
	    info.Skipped = 0;
	    info.SelectedPred = nan;
	    info.SelectedNovelty = nan;
	    info.SelectedIsMixed = false;
	end
	
	function stats = InitializeMixLSStats()
	    stats.Attempts = 0;
	    stats.BaselineCandidates = 0;
	    stats.DiscreteCandidates = 0;
	    stats.TotalCandidates = 0;
	    stats.SelectedBaseline = 0;
	    stats.SelectedMixed = 0;
	    stats.Skipped = 0;
	    stats.SelectedPred = [];
	    stats.SelectedNovelty = [];
	    stats.SelectedIsMixed = [];
	end
	
	function stats = UpdateMixLSStats(stats,info)
	    stats.Attempts = stats.Attempts + info.Attempts;
	    stats.BaselineCandidates = stats.BaselineCandidates + info.BaselineCandidates;
	    stats.DiscreteCandidates = stats.DiscreteCandidates + info.DiscreteCandidates;
	    stats.TotalCandidates = stats.TotalCandidates + info.TotalCandidates;
	    stats.SelectedBaseline = stats.SelectedBaseline + info.SelectedBaseline;
	    stats.SelectedMixed = stats.SelectedMixed + info.SelectedMixed;
	    stats.Skipped = stats.Skipped + info.Skipped;
	    if isfinite(info.SelectedPred)
	        stats.SelectedPred(end+1,1) = info.SelectedPred;
	        stats.SelectedNovelty(end+1,1) = info.SelectedNovelty;
	        stats.SelectedIsMixed(end+1,1) = info.SelectedIsMixed;
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

	function RunSummary = FinalizeLogs(History,EvalLog,SurLog,MixLSStats,runStart,Problem,historyFile)
	    RunSummary = BuildRunSummary(History,EvalLog,SurLog,MixLSStats,runStart,Problem);
	    save(historyFile,'History','EvalLog','SurLog','RunSummary');
	    PrintRunSummary(RunSummary,historyFile);
	end

	function RunSummary = BuildRunSummary(~,EvalLog,SurLog,MixLSStats,runStart,Problem)
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
	    RunSummary.MixLSAttempts = MixLSStats.Attempts;
	    RunSummary.MixLSBaselineCandidates = MixLSStats.BaselineCandidates;
	    RunSummary.MixLSDiscreteCandidates = MixLSStats.DiscreteCandidates;
	    RunSummary.MixLSTotalCandidates = MixLSStats.TotalCandidates;
	    RunSummary.MixLSSelectedBaseline = MixLSStats.SelectedBaseline;
	    RunSummary.MixLSSelectedMixed = MixLSStats.SelectedMixed;
	    RunSummary.MixLSSkipped = MixLSStats.Skipped;
	    RunSummary.MixLSMeanSelectedPred = mean(MixLSStats.SelectedPred,'omitnan');
	    RunSummary.MixLSMeanSelectedNovelty = mean(MixLSStats.SelectedNovelty,'omitnan');
	    RunSummary.MixLSMixedSelectionRate = mean(MixLSStats.SelectedIsMixed,'omitnan');

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
	    fprintf('\n========== MiSACOMixLS Run Summary ==========\n');
    fprintf('Best Objective: %.12g\n',RunSummary.BestObjective);
    fprintf('Best FE: %d\n',RunSummary.BestFE);
    fprintf('Runtime: %.2fs\n',RunSummary.Runtime);
    fprintf('Total FE: %d\n\n',RunSummary.TotalFE);

	    fprintf('RBF Evaluations: %d\n',RunSummary.RBFEvaluations);
	    fprintf('RF Evaluations: %d\n',RunSummary.RFEvaluations);
	    fprintf('Random Evaluations: %d\n',RunSummary.RandomEvaluations);
	    fprintf('LocalSearch Evaluations: %d\n\n',RunSummary.LocalSearchEvaluations);
	    fprintf('MixLS attempts: %d\n',RunSummary.MixLSAttempts);
	    fprintf('MixLS baseline candidates: %d\n',RunSummary.MixLSBaselineCandidates);
	    fprintf('MixLS discrete candidates: %d\n',RunSummary.MixLSDiscreteCandidates);
	    fprintf('MixLS selected baseline: %d\n',RunSummary.MixLSSelectedBaseline);
	    fprintf('MixLS selected mixed: %d\n',RunSummary.MixLSSelectedMixed);
	    fprintf('MixLS skipped: %d\n',RunSummary.MixLSSkipped);
	    fprintf('MixLS mixed selection rate: %.4g\n\n',RunSummary.MixLSMixedSelectionRate);

    fprintf('RBF Improvements: %d\n',RunSummary.RBFImprovements);
    fprintf('RF Improvements: %d\n',RunSummary.RFImprovements);
    fprintf('Random Improvements: %d\n',RunSummary.RandomImprovements);
    fprintf('LocalSearch Improvements: %d\n\n',RunSummary.LocalSearchImprovements);

    fprintf('RBF MAE: %.12g, RBF RMSE: %.12g\n',RunSummary.RBF_MAE,RunSummary.RBF_RMSE);
    fprintf('RF MAE: %.12g, RF RMSE: %.12g\n',RunSummary.RF_MAE,RunSummary.RF_RMSE);
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('========================================\n\n');
end
