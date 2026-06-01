classdef AGESV3 < ALGORITHM
% <single> <real> <expensive>
% Unified ensemble AGES for EOPCCV.
% Np              --- 60   --- Size of parent population
% No              --- 100  --- Size of offspring population
% TopK            --- 25   --- Candidate shortlist size for unified scoring
% MinAlign        --- 60   --- Minimum alignment samples before using the model
% Lambda          --- 0.10 --- Soft penalty weight for predicted surrogate error
% Beta            --- 0.25 --- Reward weight for mixed novelty
% Gamma           --- 0.05 --- Reward weight for surrogate disagreement
% VetoQuantile    --- 0.85 --- Error-rank veto threshold inside the shortlist
% BatchSize       --- 2    --- Number of unified candidates evaluated per loop
% DiversityWeight --- 0.10 --- Diversity reward for the second batch point
% ModelCycles     --- 30   --- Tree ensemble learning cycles

    methods
        function main(Algorithm,Problem)
            [History,EvalLog,SurLog,AlignLog] = InitializeLogs();
            AlignDB     = InitializeAlignmentDB();
            currentBest = inf;
            runStart    = tic;
            algorithmFile = which(class(Algorithm));
            historyFile = getenv('AGESV3_HISTORY_FILE');
            if isempty(historyFile)
                historyFile = getenv('AGESV2_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = getenv('AGES_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = getenv('MISACO_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = fullfile(fileparts(fileparts(algorithmFile)),'history.AGESV3.mat');
            end

            try
                [Np,No,TopK,MinAlign,Lambda,Beta,Gamma,VetoQuantile,BatchSize,DiversityWeight,ModelCycles] = ...
                    Algorithm.ParameterSet(60,100,25,60,0.10,0.25,0.05,0.85,2,0.10,30);
                TopK            = max(1,round(TopK));
                MinAlign        = max(0,round(MinAlign));
                Lambda          = max(0,Lambda);
                Beta            = max(0,Beta);
                Gamma           = max(0,Gamma);
                VetoQuantile    = min(max(VetoQuantile,0.50),0.95);
                BatchSize       = min(2,max(1,round(BatchSize)));
                DiversityWeight = max(0,DiversityWeight);
                ModelCycles     = max(5,round(ModelCycles));

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

                    [~,rank_v] = sort(P.objs);
                    [~,rank_v] = sort(rank_v);
                    OffDec = ACO_MV_generate(P.decs,Problem.Dr,Problem.Dc,Np,NormalizeLenC(Problem.len_c,Problem.Dc),rank_v,No,upper_r,lower_r);

                    Surrogates = TrainSurrogatesAGESV3( ...
                        Train_X,Train_Y,Problem,upper_r,lower_r,ModelCycles);
                    Pred = PredictSurrogatesAGESV3( ...
                        Surrogates,OffDec,Problem,upper_r,lower_r);

                    [ids_selected,features_selected,ensemblePred_selected,alignInfos] = ...
                        SelectBatchAGESV3( ...
                            OffDec,Pred,Train_X,Train_Y,Problem,upper_r,lower_r, ...
                            TopK,MinAlign,Lambda,Beta,Gamma,VetoQuantile, ...
                            BatchSize,DiversityWeight,currentBest,AlignDB);

                    for kSelect = 1 : numel(ids_selected)
                        id_selected = ids_selected(kSelect);
                        feBefore = Problem.FE;
                        New      = Problem.Evaluation(OffDec(id_selected,:));
                        SurLog   = LogSurrogatePredictions( ...
                            SurLog,Surrogates.Names,Pred(id_selected,:), ...
                            ensemblePred_selected(kSelect),New.objs,Problem.FE);
                        [AlignDB,AlignLog] = LogAlignmentEvaluation( ...
                            AlignDB,AlignLog,features_selected(kSelect,:), ...
                            'Ensemble',ensemblePred_selected(kSelect), ...
                            New.objs,Problem.FE,alignInfos(kSelect));
                        [History,EvalLog,currentBest] = LogRealEvaluation( ...
                            History,EvalLog,New,'Ensemble',feBefore,currentBest);
                        DB = [DB, New];
                    end

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

function Surrogates = TrainSurrogatesAGESV3(Train_X,Train_Y,Problem,upper_r,lower_r,ModelCycles)
    catIdx = Problem.Dr+1:Problem.D;
    y = Train_Y(:,1);
    treeMinLeaf = max(1,ceil(size(Train_X,1)*0.02));

    Surrogates.Names = {'RBF','RF','ET','GBDT'};
    Surrogates.RBF = [];
    Surrogates.RF = [];
    Surrogates.ET = [];
    Surrogates.GBDT = [];

    try
        Surrogates.RBF = RBFCreate_Hamming( ...
            Train_X,y,Problem.Dc,upper_r,lower_r,'gaussian');
    catch
        Surrogates.RBF = [];
    end

    try
        tree = templateTree('MinLeafSize',treeMinLeaf);
        Surrogates.RF = fitrensemble( ...
            Train_X,y, ...
            'Method','Bag', ...
            'NumLearningCycles',ModelCycles, ...
            'Learners',tree, ...
            'CategoricalPredictors',catIdx);
    catch
        Surrogates.RF = [];
    end

    try
        numPredictors = max(1,ceil(sqrt(Problem.D)));
        Surrogates.ET = TreeBagger( ...
            ModelCycles,Train_X,y, ...
            'Method','regression', ...
            'CategoricalPredictors',catIdx, ...
            'NumPredictorsToSample',numPredictors, ...
            'MinLeafSize',1, ...
            'SampleWithReplacement','on');
    catch
        Surrogates.ET = [];
    end

    try
        tree = templateTree('MinLeafSize',treeMinLeaf);
        Surrogates.GBDT = fitrensemble( ...
            Train_X,y, ...
            'Method','LSBoost', ...
            'NumLearningCycles',ModelCycles, ...
            'LearnRate',0.05, ...
            'Learners',tree, ...
            'CategoricalPredictors',catIdx);
    catch
        Surrogates.GBDT = [];
    end
end

function Pred = PredictSurrogatesAGESV3(Surrogates,OffDec,Problem,upper_r,lower_r)
    n = size(OffDec,1);
    Pred = nan(n,numel(Surrogates.Names));

    if ~isempty(Surrogates.RBF)
        try
            for i = 1:n
                Pred(i,1) = RBFInterp_Hamming( ...
                    OffDec(i,:),Problem.Dc,upper_r,lower_r,Surrogates.RBF);
            end
        catch
            Pred(:,1) = nan;
        end
    end

    if ~isempty(Surrogates.RF)
        Pred(:,2) = SafePredict(Surrogates.RF,OffDec,n);
    end
    if ~isempty(Surrogates.ET)
        Pred(:,3) = SafePredict(Surrogates.ET,OffDec,n);
    end
    if ~isempty(Surrogates.GBDT)
        Pred(:,4) = SafePredict(Surrogates.GBDT,OffDec,n);
    end
end

function pred = SafePredict(model,X,n)
    pred = nan(n,1);
    try
        raw = predict(model,X);
        if iscell(raw)
            raw = str2double(raw);
        end
        pred = raw(:);
        if numel(pred) ~= n
            pred = nan(n,1);
        end
        pred(~isfinite(pred)) = nan;
    catch
        pred = nan(n,1);
    end
end

function [idsSelected,featuresSelected,ensemblePredSelected,infos] = SelectBatchAGESV3( ...
    OffDec,Pred,Train_X,Train_Y,Problem,upper_r,lower_r, ...
    TopK,MinAlign,Lambda,Beta,Gamma,VetoQuantile,BatchSize,DiversityWeight,currentBest,AlignDB)

    n = size(OffDec,1);
    [Features,FeatureInfo] = BuildAlignmentFeatures( ...
        OffDec,Pred,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest);

    ensemblePred = FeatureInfo.EnsemblePred;
    candidateIdx = Shortlist(ensemblePred,TopK);
    if isempty(candidateIdx)
        candidateIdx = (1:n)';
    end

    useAlignment = false;
    predLogError = nan(n,1);
    errorPenalty = zeros(n,1);
    vetoed = false(numel(candidateIdx),1);
    activeIdx = candidateIdx;

    if HasAlignmentData(AlignDB,MinAlign)
        [predLogError,modelOK] = PredictAlignmentError(AlignDB,Features);
        if modelOK
            useAlignment = true;
            errorPenalty = NormalizedRank(predLogError);
            shortlistPenalty = errorPenalty(candidateIdx);
            vetoThreshold = QuantileFinite(shortlistPenalty,VetoQuantile);
            vetoed = isfinite(shortlistPenalty) & shortlistPenalty >= vetoThreshold;
            activeIdx = candidateIdx(~vetoed);
            if isempty(activeIdx)
                activeIdx = candidateIdx;
                vetoed = false(size(candidateIdx));
            end
        end
    end

    score = FeatureInfo.Quality + ...
            Beta*FeatureInfo.MixedNovelty + ...
            Gamma*FeatureInfo.Disagreement - ...
            Lambda*errorPenalty;
    score(~isfinite(score)) = -inf;
    if all(isinf(score(activeIdx)))
        score = FeatureInfo.MixedNovelty;
        score(~isfinite(score)) = 0;
    end

    idsSelected = zeros(1,min(BatchSize,numel(activeIdx)));
    available = activeIdx(:);
    for k = 1:numel(idsSelected)
        adjustedScore = score(available);
        if k > 1 && DiversityWeight > 0
            distanceToSelected = DistanceToSelected( ...
                OffDec(available,:),OffDec(idsSelected(1:k-1),:),Problem,upper_r,lower_r);
            adjustedScore = adjustedScore + DiversityWeight*UnitNormalize(distanceToSelected);
        end
        [~,bestLocal] = max(adjustedScore);
        idsSelected(k) = available(bestLocal);
        available(bestLocal) = [];
    end

    featuresSelected = Features(idsSelected,:);
    ensemblePredSelected = ensemblePred(idsSelected);
    infos = repmat(DefaultAlignInfo(),numel(idsSelected),1);
    for k = 1:numel(idsSelected)
        idx = idsSelected(k);
        infos(k).UsedAlignment = useAlignment;
        infos(k).PredLogError  = predLogError(idx);
        infos(k).PredAbsError  = exp(predLogError(idx)) - 1;
        infos(k).Score         = score(idx);
        infos(k).Quality       = FeatureInfo.Quality(idx);
        infos(k).Novelty       = FeatureInfo.MixedNovelty(idx);
        infos(k).VetoedCandidates = sum(vetoed);
    end
end

function [Features,Info] = BuildAlignmentFeatures( ...
    OffDec,Pred,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest)

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

    predRank = nan(size(Pred));
    for j = 1:size(Pred,2)
        predRank(:,j) = NormalizedRank(Pred(:,j));
    end
    ensemblePred = RowMedianFinite(Pred);
    predMean = RowMeanFinite(Pred);
    predStd = RowStdFinite(Pred);
    voteRank = RowMeanFinite(predRank);
    modelCount = sum(isfinite(Pred),2)./size(Pred,2);

    predImprovement = currentBest - ensemblePred;
    [nearestDist,avgKDist,sameCatCount] = CandidateDistances(OffDec,Train_X,Problem,upper_r,lower_r);
    distanceNovelty = UnitNormalize(nearestDist);
    ensembleQuality = 1 - NormalizedRank(ensemblePred);
    voteQuality = 1 - voteRank;
    quality = 0.65*ensembleQuality + 0.35*voteQuality;
    quality(~isfinite(quality)) = 0;

    bestNorm = NormalizeWithReference(currentBest,Train_Y);
    sameCatSeen = sameCatCount > 0;
    sameCatScaled = min(1,sameCatCount./max(1,size(Train_X,1)));
    categoryNovelty = 1 - sameCatScaled;
    categoryWeight = Problem.Dc./max(1,Problem.D);
    mixedNovelty = (1 - categoryWeight).*distanceNovelty + categoryWeight.*categoryNovelty;
    disagreement = UnitNormalize(RowStdFinite(predRank));

    Features = [ ...
        x_r, ...
        x_c_scaled, ...
        UnitNormalizeColumns(Pred), ...
        predRank, ...
        UnitNormalize(ensemblePred), ...
        UnitNormalize(predMean), ...
        UnitNormalize(predStd), ...
        disagreement, ...
        UnitNormalize(nearestDist), ...
        UnitNormalize(avgKDist), ...
        double(sameCatSeen), ...
        sameCatScaled, ...
        categoryNovelty, ...
        mixedNovelty, ...
        repmat(bestNorm,n,1), ...
        UnitNormalize(predImprovement), ...
        modelCount];
    Features = ReplaceNonFinite(Features,0.5);

    Info.Quality = quality;
    Info.Novelty = distanceNovelty;
    Info.CategoryNovelty = categoryNovelty;
    Info.MixedNovelty = mixedNovelty;
    Info.Disagreement = disagreement;
    Info.EnsemblePred = ensemblePred;
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

function distance = DistanceToSelected(Candidates,Selected,Problem,upper_r,lower_r)
    n = size(Candidates,1);
    distance = zeros(n,1);
    if isempty(Selected)
        return;
    end

    Dr = Problem.Dr;
    xrange = upper_r - lower_r;
    xrange(xrange < eps) = eps;
    candR = (Candidates(:,1:Dr) - lower_r)./xrange;
    selR = (Selected(:,1:Dr) - lower_r)./xrange;
    candC = ceil(Candidates(:,Dr+1:Problem.D));
    selC = ceil(Selected(:,Dr+1:Problem.D));

    for i = 1:n
        distR = mean(abs(selR - candR(i,:)),2);
        distC = mean(selC ~= candC(i,:),2);
        distance(i) = min(0.5*distR + 0.5*distC);
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

function Xn = UnitNormalizeColumns(X)
    Xn = zeros(size(X));
    for j = 1:size(X,2)
        Xn(:,j) = UnitNormalize(X(:,j));
    end
end

function out = RowMedianFinite(X)
    out = nan(size(X,1),1);
    for i = 1:size(X,1)
        row = X(i,:);
        row = row(isfinite(row));
        if ~isempty(row)
            out(i) = median(row);
        end
    end
end

function out = RowMeanFinite(X)
    out = nan(size(X,1),1);
    for i = 1:size(X,1)
        row = X(i,:);
        row = row(isfinite(row));
        if ~isempty(row)
            out(i) = mean(row);
        end
    end
end

function out = RowStdFinite(X)
    out = zeros(size(X,1),1);
    for i = 1:size(X,1)
        row = X(i,:);
        row = row(isfinite(row));
        if numel(row) > 1
            out(i) = std(row);
        end
    end
end

function q = QuantileFinite(values,probability)
    values = sort(values(isfinite(values)));
    if isempty(values)
        q = inf;
        return;
    end
    probability = min(max(probability,0),1);
    idx = max(1,min(numel(values),ceil(probability*numel(values))));
    q = values(idx);
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
    info.VetoedCandidates = 0;
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

function SurLog = LogSurrogatePredictions(SurLog,modelNames,predictions,ensemblePrediction,realValue,fe)
    realValue = realValue(1);
    for i = 1:numel(modelNames)
        SurLog.FE(end+1,1)     = fe;
        SurLog.Source{end+1,1} = modelNames{i};
        SurLog.Pred(end+1,1)   = predictions(i);
        SurLog.Real(end+1,1)   = realValue;
    end

    SurLog.FE(end+1,1)     = fe;
    SurLog.Source{end+1,1} = 'Ensemble';
    SurLog.Pred(end+1,1)   = ensemblePrediction(1);
    SurLog.Real(end+1,1)   = realValue;
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
    sources = {'Ensemble','RBF','RF','Random','LocalSearch'};

    if isempty(EvalLog.Obj)
        bestObj = nan;
        bestFE  = nan;
    else
        [bestObj,idBest] = min(EvalLog.Obj);
        bestFE = EvalLog.FE(idBest);
    end

    RunSummary.Algorithm     = 'AGESV3';
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
    [RunSummary.ET_MAE, RunSummary.ET_RMSE]  = SurrogateError(SurLog,'ET');
    [RunSummary.GBDT_MAE, RunSummary.GBDT_RMSE] = SurrogateError(SurLog,'GBDT');
    [RunSummary.Ensemble_MAE, RunSummary.Ensemble_RMSE] = SurrogateError(SurLog,'Ensemble');
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
    fprintf('\n========== AGESV3 Run Summary ==========\n');
    fprintf('Best Objective: %.12g\n',RunSummary.BestObjective);
    fprintf('Best FE: %d\n',RunSummary.BestFE);
    fprintf('Runtime: %.2fs\n',RunSummary.Runtime);
    fprintf('Total FE: %d\n\n',RunSummary.TotalFE);

    fprintf('Ensemble Evaluations: %d\n',RunSummary.EnsembleEvaluations);
    fprintf('RBF Evaluations: %d\n',RunSummary.RBFEvaluations);
    fprintf('RF Evaluations: %d\n',RunSummary.RFEvaluations);
    fprintf('Random Evaluations: %d\n',RunSummary.RandomEvaluations);
    fprintf('LocalSearch Evaluations: %d\n\n',RunSummary.LocalSearchEvaluations);

    fprintf('Ensemble Improvements: %d\n',RunSummary.EnsembleImprovements);
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
    fprintf('ET MAE: %.12g, ET RMSE: %.12g\n',RunSummary.ET_MAE,RunSummary.ET_RMSE);
    fprintf('GBDT MAE: %.12g, GBDT RMSE: %.12g\n',RunSummary.GBDT_MAE,RunSummary.GBDT_RMSE);
    fprintf('Ensemble MAE: %.12g, Ensemble RMSE: %.12g\n',RunSummary.Ensemble_MAE,RunSummary.Ensemble_RMSE);
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('======================================\n\n');
end
