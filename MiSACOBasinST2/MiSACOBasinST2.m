classdef MiSACOBasinST2 < ALGORITHM
% <single> <real> <expensive>
% MiSACO with Rastrigin-only stagnation-triggered cross-basin exploration.
% Np                 --- 60   --- Size of parent population
% No                 --- 100  --- Size of offspring population
% BasinPool          --- 120  --- Number of generated basin-hop candidates
% BasinTopK          --- 30   --- Surrogate-quality shortlist size
% BasinNoveltyWeight --- 0.25 --- Weight for mixed decision novelty
% BasinArchiveWeight --- 0.35 --- Weight for unseen basin signatures
% DisagreementWeight --- 0.10 --- Weight for RBF/RF disagreement
% MinDestroyRatio    --- 0.25 --- Minimum variable fraction changed by hop
% MaxDestroyRatio    --- 0.60 --- Maximum variable fraction changed by hop
% RastriginStagFE    --- 24   --- FE stagnation threshold on Rastrigin cases
% EasyTolerance      --- 1e-3 --- Disable BasinHop when best is already small
% PredImprovementTol --- 0.02 --- Minimum predicted relative improvement

    methods
        function main(Algorithm,Problem)
            [History,EvalLog,SurLog] = InitializeLogs();
            Ablation = ReadBasinAblation();
            BasinState = InitializeBasinState(Ablation);
            TriggerState = InitializeTriggerState(Problem,24,1e-3,0.02);
            currentBest = inf;
            runStart    = tic;
            algorithmFile = which(class(Algorithm));
            historyFile = getenv('MISACOBASINST2_HISTORY_FILE');
            if isempty(historyFile)
                historyFile = getenv('MISACOBASINST_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = getenv('MISACOBASIN_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = getenv('MISACO_HISTORY_FILE');
            end
            if isempty(historyFile)
                historyFile = fullfile(fileparts(fileparts(algorithmFile)),'history.MiSACOBasinST2.mat');
            end

            try
                [Np,No,BasinPool,BasinTopK,BasinNoveltyWeight,BasinArchiveWeight, ...
                    DisagreementWeight,MinDestroyRatio,MaxDestroyRatio,RastriginStagFE, ...
                    EasyTolerance,PredImprovementTol] = ...
                    Algorithm.ParameterSet(60,100,120,30,0.25,0.35,0.10,0.25,0.60,24,1e-3,0.02);
                BasinPool = max(20,round(BasinPool));
                BasinTopK = max(1,round(BasinTopK));
                BasinNoveltyWeight = max(0,BasinNoveltyWeight);
                BasinArchiveWeight = max(0,BasinArchiveWeight);
                DisagreementWeight = max(0,DisagreementWeight);
                MinDestroyRatio = min(max(MinDestroyRatio,0.05),0.90);
                MaxDestroyRatio = min(max(MaxDestroyRatio,MinDestroyRatio),0.95);
                TriggerState = InitializeTriggerState( ...
                    Problem,RastriginStagFE,EasyTolerance,PredImprovementTol);

                PI       = UniformPoint(Np,Problem.D,'Latin');
                feBefore = Problem.FE;
                DB       = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Np,1).*PI+repmat(Problem.lower,Np,1));
                [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,DB,'Init',feBefore,currentBest);
                TriggerState = UpdateTriggerState(TriggerState,inf,currentBest,Problem.FE);

                P.decs = DB.decs;
                P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                P.objs = DB.objs;

                while Algorithm.NotTerminated(DB)
                    Train_X = DB.decs;
                    Train_X(:,Problem.Dr+1:Problem.D) = ceil(Train_X(:,Problem.Dr+1:Problem.D));
                    Train_Y = DB.objs;
                    upper_r = Problem.upper(:,1:Problem.Dr);
                    lower_r = Problem.lower(:,1:Problem.Dr);
                    lenC    = NormalizeLenC(Problem.len_c,Problem.Dc);

                    RBF_glo = RBFCreate_Hamming(Train_X,Train_Y,Problem.Dc,upper_r,lower_r,'gaussian');
                    RF_net  = fitrensemble(Train_X,Train_Y,'CategoricalPredictors',Problem.Dr+1:Problem.D);

                    [~,rank_v] = sort(P.objs);
                    [~,rank_v] = sort(rank_v);
                    OffDec = ACO_MV_generate(P.decs,Problem.Dr,Problem.Dc,Np,lenC,rank_v,No,upper_r,lower_r);

                    RBF_evaluate = zeros(size(OffDec,1),1);
                    for i = 1 : size(OffDec,1)
                        RBF_evaluate(i) = RBFInterp_Hamming(OffDec(i,:),Problem.Dc,upper_r,lower_r,RBF_glo);
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
                    feBefore = Problem.FE;
                    bestBefore = currentBest;
                    New      = Problem.Evaluation(OffDec(id_selected,:));
                    SurLog   = LogSurrogatePrediction(SurLog,'RBF',pred_rbf,New.objs,Problem.FE);
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RBF',feBefore,currentBest);
                    TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];

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
                    feBefore = Problem.FE;
                    bestBefore = currentBest;
                    New      = Problem.Evaluation(OffDec(id_selected,:));
                    SurLog   = LogSurrogatePrediction(SurLog,'RF',pred_rf,New.objs,Problem.FE);
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RF',feBefore,currentBest);
                    TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];

                    useBasinHop = ShouldTriggerBasinHop(TriggerState,currentBest,Problem.FE);
                    if useBasinHop
                        TriggerState = RecordBasinTriggerAttempt(TriggerState);
                        [xBasin,pred_basin,opId,acceptedByVeto] = SelectBasinHopCandidate( ...
                            DB,Train_X,Train_Y,RBF_glo,RF_net,Problem,upper_r,lower_r,lenC, ...
                            BasinState,BasinPool,BasinTopK,BasinNoveltyWeight,BasinArchiveWeight, ...
                            DisagreementWeight,MinDestroyRatio,MaxDestroyRatio,currentBest,Ablation,TriggerState);
                        if acceptedByVeto
                            feBefore = Problem.FE;
                            bestBefore = currentBest;
                            New      = Problem.Evaluation(xBasin);
                            SurLog   = LogSurrogatePrediction(SurLog,'BasinHop',pred_basin,New.objs,Problem.FE);
                            [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'BasinHop',feBefore,currentBest);
                            TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                            TriggerState = RecordBasinTrigger(TriggerState,Problem.FE);
                            BasinState = UpdateBasinState(BasinState,xBasin,New.objs,bestBefore,opId,Problem,upper_r,lower_r,Ablation);
                            DB = [DB, New];
                        else
                            TriggerState = RecordBasinVeto(TriggerState,Problem.FE);
                            id_selected = randi(size(OffDec,1));
                            feBefore = Problem.FE;
                            bestBefore = currentBest;
                            New = Problem.Evaluation(OffDec(id_selected,:));
                            [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'Random',feBefore,currentBest);
                            TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                            TriggerState.RandomFallbacks = TriggerState.RandomFallbacks + 1;
                            DB = [DB, New];
                        end
                    else
                        id_selected = randi(size(OffDec,1));
                        feBefore = Problem.FE;
                        bestBefore = currentBest;
                        New = Problem.Evaluation(OffDec(id_selected,:));
                        [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'Random',feBefore,currentBest);
                        TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                        TriggerState.RandomFallbacks = TriggerState.RandomFallbacks + 1;
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
                            bestBefore = currentBest;
                            New      = Problem.Evaluation([xr_Local,best_c]);
                            [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'LocalSearch',feBefore,currentBest);
                            TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                            DB = [DB, New];
                        end
                    end

                    [~,rank] = sort(DB.objs);
                    P.decs = DB(rank(1:Np)).decs;
                    P.decs(:,Problem.Dr+1:Problem.D) = ceil(P.decs(:,Problem.Dr+1:Problem.D));
                    P.objs = DB(rank(1:Np)).objs;
                end
            catch err
                FinalizeLogs(History,EvalLog,SurLog,BasinState,TriggerState,runStart,Problem,historyFile);
                rethrow(err);
            end
            FinalizeLogs(History,EvalLog,SurLog,BasinState,TriggerState,runStart,Problem,historyFile);
        end
    end
end

function [xSelected,predSelected,opSelected,acceptedByVeto] = SelectBasinHopCandidate( ...
    DB,Train_X,Train_Y,RBF_glo,RF_net,Problem,upper_r,lower_r,lenC,BasinState, ...
    BasinPool,BasinTopK,BasinNoveltyWeight,BasinArchiveWeight,DisagreementWeight, ...
    MinDestroyRatio,MaxDestroyRatio,currentBest,Ablation,TriggerState)

    [Candidates,OpIds] = GenerateBasinHopCandidates( ...
        DB,Problem,upper_r,lower_r,lenC,BasinState,BasinPool,MinDestroyRatio,MaxDestroyRatio);

    if Ablation.RandomHopSelection
        idSelected = randi(size(Candidates,1));
        xSelected = Candidates(idSelected,:);
        predSelected = nan;
        opSelected = OpIds(idSelected);
        acceptedByVeto = true;
        return;
    end

    RBFPred = PredictRBF(Candidates,RBF_glo,Problem,upper_r,lower_r);
    RFPred  = SafePredict(RF_net,Candidates);
    predMat = [RBFPred(:),RFPred(:)];
    ensemblePred = MeanFiniteRows(predMat);

    quality = PredictionQuality(ensemblePred,Train_Y,currentBest);
    mixedNovelty = MixedNearestNovelty(Candidates,Train_X,Problem,upper_r,lower_r);
    if Ablation.UseArchive
        basinNovelty = BasinArchiveNovelty(Candidates,Train_X,BasinState,Problem,upper_r,lower_r);
    else
        basinNovelty = zeros(size(Candidates,1),1);
        BasinArchiveWeight = 0;
    end
    disagreement = UnitNormalize(abs(RBFPred(:) - RFPred(:)));

    qualityIdx = ShortlistByPrediction(ensemblePred,BasinTopK);
    noveltyIdx = TopKIndices(mixedNovelty + basinNovelty,max(1,ceil(BasinTopK/2)));
    candidateIdx = unique([qualityIdx(:); noveltyIdx(:)]);
    if isempty(candidateIdx)
        candidateIdx = (1:size(Candidates,1))';
    end

    score = quality + ...
        BasinNoveltyWeight*mixedNovelty + ...
        BasinArchiveWeight*basinNovelty + ...
        DisagreementWeight*disagreement;
    score(~isfinite(score)) = -inf;
    [~,localBest] = max(score(candidateIdx));
    idSelected = candidateIdx(localBest);

    xSelected = Candidates(idSelected,:);
    predSelected = ensemblePred(idSelected);
    opSelected = OpIds(idSelected);
    acceptedByVeto = PassPredictionVeto(predSelected,currentBest,TriggerState.PredImprovementTol);
end

function accepted = PassPredictionVeto(predicted,currentBest,predImprovementTol)
    if ~isfinite(predicted) || ~isfinite(currentBest)
        accepted = false;
        return;
    end

    predictedImprovement = currentBest - predicted;
    relativeImprovement = predictedImprovement/max(1,abs(currentBest));
    accepted = predictedImprovement > 0 && relativeImprovement >= predImprovementTol;
end

function [Candidates,OpIds] = GenerateBasinHopCandidates( ...
    DB,Problem,upper_r,lower_r,lenC,BasinState,BasinPool,MinDestroyRatio,MaxDestroyRatio)

    D  = Problem.D;
    Dr = Problem.Dr;
    Dc = Problem.Dc;
    dbDecs = DB.decs;
    dbDecs(:,Dr+1:D) = ceil(dbDecs(:,Dr+1:D));
    [~,rank] = sort(DB.objs);
    eliteCount = min(12,numel(rank));
    elites = dbDecs(rank(1:eliteCount),:);
    periods = BasinPeriods(upper_r,lower_r);

    Candidates = zeros(BasinPool,D);
    OpIds = zeros(BasinPool,1);
    for i = 1 : BasinPool
        opId = RouletteWheel(BasinState.OperatorWeights);
        base = elites(SampleElite(eliteCount),:);
        x = base;

        switch opId
            case 1
                x = ApplyPeriodHop(x,Dr,Dc,upper_r,lower_r,lenC,periods,MinDestroyRatio,MaxDestroyRatio);
            case 2
                x = ApplyWideRestart(x,Dr,Dc,upper_r,lower_r,lenC,MinDestroyRatio,MaxDestroyRatio);
            case 3
                x = ApplyCategoricalHop(x,Dr,Dc,upper_r,lower_r,lenC,periods,MinDestroyRatio,MaxDestroyRatio);
            otherwise
                mate = elites(SampleElite(eliteCount),:);
                x = ApplyEliteRecombine(x,mate,Dr,Dc,upper_r,lower_r,lenC,periods,MinDestroyRatio,MaxDestroyRatio);
        end

        Candidates(i,:) = RepairMixed(x,Dr,Dc,upper_r,lower_r,lenC);
        OpIds(i) = opId;
    end
end

function x = ApplyPeriodHop(x,Dr,Dc,upper_r,lower_r,lenC,periods,MinDestroyRatio,MaxDestroyRatio)
    if Dr > 0
        nChange = MutationCount(Dr,MinDestroyRatio,MaxDestroyRatio);
        dims = randperm(Dr,nChange);
        hops = randi([1,3],1,nChange) .* (2*(rand(1,nChange) > 0.5)-1);
        jitter = 0.08*periods(dims).*randn(1,nChange);
        x(dims) = x(dims) + hops.*periods(dims) + jitter;
    end

    if Dc > 0 && rand < 0.25
        x = MutateCategoricalPart(x,Dr,Dc,lenC,0.10,0.30);
    end
    x = RepairMixed(x,Dr,Dc,upper_r,lower_r,lenC);
end

function x = ApplyWideRestart(x,Dr,Dc,upper_r,lower_r,lenC,MinDestroyRatio,MaxDestroyRatio)
    if Dr > 0
        nChange = MutationCount(Dr,max(MinDestroyRatio,0.40),MaxDestroyRatio);
        dims = randperm(Dr,nChange);
        x(dims) = lower_r(dims) + rand(1,nChange).*(upper_r(dims)-lower_r(dims));
    end

    if Dc > 0
        x = MutateCategoricalPart(x,Dr,Dc,lenC,max(MinDestroyRatio,0.35),MaxDestroyRatio);
    end
    x = RepairMixed(x,Dr,Dc,upper_r,lower_r,lenC);
end

function x = ApplyCategoricalHop(x,Dr,Dc,upper_r,lower_r,lenC,periods,MinDestroyRatio,MaxDestroyRatio)
    if Dc > 0
        x = MutateCategoricalPart(x,Dr,Dc,lenC,MinDestroyRatio,MaxDestroyRatio);
    end

    if Dr > 0
        nChange = max(1,ceil(0.20*Dr));
        dims = randperm(Dr,nChange);
        hops = (2*(rand(1,nChange) > 0.5)-1);
        x(dims) = x(dims) + hops.*periods(dims) + 0.05*periods(dims).*randn(1,nChange);
    end
    x = RepairMixed(x,Dr,Dc,upper_r,lower_r,lenC);
end

function x = ApplyEliteRecombine(x,mate,Dr,Dc,upper_r,lower_r,lenC,periods,MinDestroyRatio,MaxDestroyRatio)
    D = Dr + Dc;
    nCopy = MutationCount(D,MinDestroyRatio,MaxDestroyRatio);
    dims = randperm(D,nCopy);
    x(dims) = mate(dims);

    if Dr > 0
        nHop = max(1,ceil(0.20*Dr));
        hopDims = randperm(Dr,nHop);
        hops = randi([1,2],1,nHop) .* (2*(rand(1,nHop) > 0.5)-1);
        x(hopDims) = x(hopDims) + hops.*periods(hopDims);
    end
    x = RepairMixed(x,Dr,Dc,upper_r,lower_r,lenC);
end

function x = MutateCategoricalPart(x,Dr,Dc,lenC,minRatio,maxRatio)
    nChange = MutationCount(Dc,minRatio,maxRatio);
    dims = randperm(Dc,nChange);
    for t = 1 : numel(dims)
        j = dims(t);
        levels = max(1,lenC(j));
        oldValue = round(x(Dr+j));
        if levels <= 1
            x(Dr+j) = 1;
        else
            newValue = randi(levels);
            if newValue == oldValue
                newValue = mod(newValue,levels) + 1;
            end
            x(Dr+j) = newValue;
        end
    end
end

function nChange = MutationCount(n,minRatio,maxRatio)
    if n <= 0
        nChange = 0;
        return;
    end
    low = max(1,ceil(minRatio*n));
    high = max(low,ceil(maxRatio*n));
    high = min(high,n);
    nChange = randi([low,high]);
end

function idx = SampleElite(eliteCount)
    if eliteCount <= 1
        idx = 1;
        return;
    end
    weights = 1./(1:eliteCount);
    idx = RouletteWheel(weights);
end

function idx = RouletteWheel(weights)
    weights = max(0,weights(:));
    if isempty(weights) || sum(weights) <= eps
        idx = randi(numel(weights));
        return;
    end
    target = rand*sum(weights);
    running = cumsum(weights);
    idx = find(running >= target,1,'first');
end

function x = RepairMixed(x,Dr,Dc,upper_r,lower_r,lenC)
    if Dr > 0
        x(1:Dr) = min(max(x(1:Dr),lower_r),upper_r);
    end
    if Dc > 0
        cat = round(x(Dr+1:Dr+Dc));
        cat = max(1,cat);
        cat = min(cat,lenC);
        x(Dr+1:Dr+Dc) = cat;
    end
end

function pred = PredictRBF(Candidates,RBF_glo,Problem,upper_r,lower_r)
    n = size(Candidates,1);
    pred = nan(n,1);
    for i = 1 : n
        try
            pred(i) = RBFInterp_Hamming(Candidates(i,:),Problem.Dc,upper_r,lower_r,RBF_glo);
        catch
            pred(i) = nan;
        end
    end
end

function pred = SafePredict(model,X)
    try
        pred = predict(model,X);
        pred = pred(:);
    catch
        pred = nan(size(X,1),1);
    end
end

function out = MeanFiniteRows(values)
    out = nan(size(values,1),1);
    for i = 1 : size(values,1)
        row = values(i,:);
        row = row(isfinite(row));
        if ~isempty(row)
            out(i) = mean(row);
        end
    end
end

function quality = PredictionQuality(pred,Train_Y,currentBest)
    pred = pred(:);
    quality = zeros(size(pred));
    finite = isfinite(pred);
    if ~any(finite)
        return;
    end

    ref = [Train_Y(:); currentBest; pred(finite)];
    ref = ref(isfinite(ref));
    if isempty(ref)
        quality(finite) = 0.5;
        return;
    end

    low = min(ref);
    high = max(ref);
    if high <= low
        quality(finite) = 0.5;
    else
        normPred = (pred(finite) - low)./(high - low);
        quality(finite) = 1 - min(max(normPred,0),1);
    end
end

function novelty = MixedNearestNovelty(Candidates,Train_X,Problem,upper_r,lower_r)
    n = size(Candidates,1);
    novelty = zeros(n,1);
    if isempty(Train_X)
        novelty(:) = 1;
        return;
    end

    Dr = Problem.Dr;
    Dc = Problem.Dc;
    scale = max(upper_r - lower_r,eps);
    trainCat = ceil(Train_X(:,Dr+1:Dr+Dc));

    for i = 1 : n
        dist = zeros(size(Train_X,1),1);
        parts = 0;
        if Dr > 0
            diffR = (Train_X(:,1:Dr) - Candidates(i,1:Dr))./scale;
            dist = dist + sqrt(mean(diffR.^2,2));
            parts = parts + 1;
        end
        if Dc > 0
            diffC = mean(trainCat ~= ceil(Candidates(i,Dr+1:Dr+Dc)),2);
            dist = dist + diffC;
            parts = parts + 1;
        end
        if parts > 0
            novelty(i) = min(dist./parts);
        end
    end
    novelty = UnitNormalize(novelty);
end

function novelty = BasinArchiveNovelty(Candidates,Train_X,BasinState,Problem,upper_r,lower_r)
    candidateSig = BasinSignature(Candidates,Problem,upper_r,lower_r);
    trainSig = BasinSignature(Train_X,Problem,upper_r,lower_r);
    if isempty(BasinState.Signatures)
        archiveSig = trainSig;
    else
        archiveSig = [trainSig; BasinState.Signatures];
    end
    n = size(candidateSig,1);
    novelty = ones(n,1);
    if isempty(archiveSig)
        return;
    end

    for i = 1 : n
        same = mean(archiveSig == candidateSig(i,:),2);
        novelty(i) = 1 - max(same);
    end
end

function sig = BasinSignature(X,Problem,upper_r,lower_r)
    if isempty(X)
        sig = zeros(0,Problem.D);
        return;
    end

    Dr = Problem.Dr;
    Dc = Problem.Dc;
    sig = zeros(size(X,1),Problem.D);
    if Dr > 0
        periods = BasinPeriods(upper_r,lower_r);
        sig(:,1:Dr) = round((X(:,1:Dr) - lower_r)./periods);
    end
    if Dc > 0
        sig(:,Dr+1:Dr+Dc) = ceil(X(:,Dr+1:Dr+Dc));
    end
end

function periods = BasinPeriods(upper_r,lower_r)
    if isempty(upper_r)
        periods = zeros(1,0);
        return;
    end
    width = upper_r - lower_r;
    periods = min(1,width/4);
    periods = max(periods,width/20);
    periods(~isfinite(periods) | periods <= 0) = 1;
end

function idx = ShortlistByPrediction(pred,topK)
    finite = isfinite(pred);
    if any(finite)
        rankValue = pred(:);
        rankValue(~finite) = inf;
        [~,rank] = sort(rankValue,'ascend');
        idx = rank(1:min(topK,numel(rank)));
    else
        idx = [];
    end
end

function idx = TopKIndices(values,topK)
    values = values(:);
    values(~isfinite(values)) = -inf;
    [~,rank] = sort(values,'descend');
    idx = rank(1:min(topK,numel(rank)));
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

function TriggerState = InitializeTriggerState(Problem,RastriginStagFE,EasyTolerance,PredImprovementTol)
    TriggerState.IsRastrigin = IsRastriginProblem(Problem);
    TriggerState.LastImprovementFE = Problem.FE;
    TriggerState.LastBasinTriggerFE = -inf;
    TriggerState.BasinTriggerAttempts = 0;
    TriggerState.BasinTriggers = 0;
    TriggerState.BasinVetoes = 0;
    TriggerState.RandomFallbacks = 0;
    TriggerState.RastriginStagFE = max(1,round(RastriginStagFE));
    TriggerState.RastriginCooldownFE = max(1,round(RastriginStagFE));
    TriggerState.EasyTolerance = max(0,EasyTolerance);
    TriggerState.PredImprovementTol = max(0,PredImprovementTol);
end

function tf = IsRastriginProblem(Problem)
    problemName = class(Problem);
    token = regexp(problemName,'EOPCCV_F(\d+)$','tokens','once');
    tf = false;
    if isempty(token)
        return;
    end
    problemId = str2double(token{1});
    tf = any(problemId == [2,7,12,17,22,27]);
end

function TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,currentFE)
    if currentBest < bestBefore
        TriggerState.LastImprovementFE = currentFE;
    end
end

function tf = ShouldTriggerBasinHop(TriggerState,currentBest,currentFE)
    if ~TriggerState.IsRastrigin || currentBest <= TriggerState.EasyTolerance
        tf = false;
        return;
    end

    stagnationFE = currentFE - TriggerState.LastImprovementFE;
    sinceLastTrigger = currentFE - TriggerState.LastBasinTriggerFE;
    tf = stagnationFE >= TriggerState.RastriginStagFE && ...
         sinceLastTrigger >= TriggerState.RastriginCooldownFE;
end

function TriggerState = RecordBasinTriggerAttempt(TriggerState)
    TriggerState.BasinTriggerAttempts = TriggerState.BasinTriggerAttempts + 1;
end

function TriggerState = RecordBasinTrigger(TriggerState,currentFE)
    TriggerState.BasinTriggers = TriggerState.BasinTriggers + 1;
    TriggerState.LastBasinTriggerFE = currentFE;
end

function TriggerState = RecordBasinVeto(TriggerState,currentFE)
    TriggerState.BasinVetoes = TriggerState.BasinVetoes + 1;
    TriggerState.LastBasinTriggerFE = currentFE;
end

function Ablation = ReadBasinAblation()
    rawMode = strtrim(getenv('MISACOBASINST2_ABLATION'));
    if isempty(rawMode)
        rawMode = strtrim(getenv('MISACOBASINST_ABLATION'));
    end
    if isempty(rawMode)
        rawMode = 'rastrigin_veto';
    end

    mode = lower(strrep(rawMode,'-','_'));
    switch mode
        case {'rastrigin_veto','st2','stagnation_triggered','st','full','none','default'}
            Ablation.Name = 'rastrigin_veto';
            Ablation.UseArchive = true;
            Ablation.AdaptiveOperators = true;
            Ablation.RandomHopSelection = false;
        case {'no_archive','noarchive'}
            Ablation.Name = 'no_archive';
            Ablation.UseArchive = false;
            Ablation.AdaptiveOperators = true;
            Ablation.RandomHopSelection = false;
        case {'fixed_op','fixedop','fixed_operator','fixed_operators'}
            Ablation.Name = 'fixed_op';
            Ablation.UseArchive = true;
            Ablation.AdaptiveOperators = false;
            Ablation.RandomHopSelection = false;
        case {'random_hop','randomhop','random_selection'}
            Ablation.Name = 'random_hop';
            Ablation.UseArchive = true;
            Ablation.AdaptiveOperators = true;
            Ablation.RandomHopSelection = true;
        otherwise
            error('MiSACOBasinST2:InvalidAblation', ...
                ['Unknown MISACOBASINST2_ABLATION=%s. Valid values are ', ...
                'rastrigin_veto, no_archive, fixed_op, random_hop.'], rawMode);
    end
end

function BasinState = InitializeBasinState(Ablation)
    BasinState.Ablation = Ablation.Name;
    BasinState.UseArchive = Ablation.UseArchive;
    BasinState.AdaptiveOperators = Ablation.AdaptiveOperators;
    BasinState.RandomHopSelection = Ablation.RandomHopSelection;
    BasinState.Signatures = [];
    BasinState.OperatorWeights = ones(1,4);
    BasinState.OperatorCalls = zeros(1,4);
    BasinState.OperatorImprovements = zeros(1,4);
    BasinState.OperatorLabels = {'PeriodHop','WideRestart','CategoricalHop','EliteRecombine'};
end

function BasinState = UpdateBasinState(BasinState,x,obj,bestBefore,opId,Problem,upper_r,lower_r,Ablation)
    if opId < 1 || opId > numel(BasinState.OperatorWeights)
        return;
    end
    improved = obj(1) < bestBefore;
    if Ablation.AdaptiveOperators
        reward = 1 + 9*double(improved);
        decay = 0.90;
        BasinState.OperatorWeights(opId) = decay*BasinState.OperatorWeights(opId) + (1-decay)*reward;
    end
    BasinState.OperatorCalls(opId) = BasinState.OperatorCalls(opId) + 1;
    BasinState.OperatorImprovements(opId) = BasinState.OperatorImprovements(opId) + double(improved);

    if ~Ablation.UseArchive
        return;
    end

    sig = BasinSignature(x,Problem,upper_r,lower_r);
    if isempty(BasinState.Signatures)
        BasinState.Signatures = sig;
    else
        BasinState.Signatures = [BasinState.Signatures; sig];
    end
    maxArchive = 200;
    if size(BasinState.Signatures,1) > maxArchive
        BasinState.Signatures = BasinState.Signatures(end-maxArchive+1:end,:);
    end
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

function idx_sel = dist_ham(x,v)
    if isempty(x)
        idx_sel = true(size(v,1),1);
        return;
    end
    distt = mean(x~=v,2);
    idx_sel = (distt == 0);
end

function out = dist_r(x0,x)
    if isempty(x0)
        out = true(size(x,1),1);
        return;
    end
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

function RunSummary = FinalizeLogs(History,EvalLog,SurLog,BasinState,TriggerState,runStart,Problem,historyFile)
    RunSummary = BuildRunSummary(History,EvalLog,SurLog,BasinState,TriggerState,runStart,Problem);
    save(historyFile,'History','EvalLog','SurLog','RunSummary');
    PrintRunSummary(RunSummary,historyFile);
end

function RunSummary = BuildRunSummary(~,EvalLog,SurLog,BasinState,TriggerState,runStart,Problem)
    sources = {'RBF','RF','Random','BasinHop','LocalSearch'};

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
    RunSummary.BasinAblation = BasinState.Ablation;
    RunSummary.BasinUseArchive = BasinState.UseArchive;
    RunSummary.BasinAdaptiveOperators = BasinState.AdaptiveOperators;
    RunSummary.BasinRandomHopSelection = BasinState.RandomHopSelection;
    RunSummary.BasinIsRastrigin = TriggerState.IsRastrigin;
    RunSummary.BasinTriggerAttempts = TriggerState.BasinTriggerAttempts;
    RunSummary.BasinTriggers = TriggerState.BasinTriggers;
    RunSummary.BasinVetoes = TriggerState.BasinVetoes;
    RunSummary.BasinRandomFallbacks = TriggerState.RandomFallbacks;
    RunSummary.BasinLastImprovementFE = TriggerState.LastImprovementFE;
    RunSummary.BasinRastriginStagFE = TriggerState.RastriginStagFE;
    RunSummary.BasinEasyTolerance = TriggerState.EasyTolerance;
    RunSummary.BasinPredImprovementTol = TriggerState.PredImprovementTol;

    for i = 1 : length(sources)
        source = sources{i};
        RunSummary.([source,'Evaluations'])  = CountSource(EvalLog,source);
        RunSummary.([source,'Improvements']) = CountImprovement(EvalLog,source);
    end

    [RunSummary.RBF_MAE,RunSummary.RBF_RMSE] = SurrogateError(SurLog,'RBF');
    [RunSummary.RF_MAE, RunSummary.RF_RMSE]  = SurrogateError(SurLog,'RF');
    [RunSummary.BasinHop_MAE,RunSummary.BasinHop_RMSE] = SurrogateError(SurLog,'BasinHop');
    RunSummary.BasinOperatorLabels = BasinState.OperatorLabels;
    RunSummary.BasinOperatorWeights = BasinState.OperatorWeights;
    RunSummary.BasinOperatorCalls = BasinState.OperatorCalls;
    RunSummary.BasinOperatorImprovements = BasinState.OperatorImprovements;
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
    fprintf('\n========== MiSACOBasinST2 Run Summary ==========\n');
    fprintf('Basin Ablation: %s\n',RunSummary.BasinAblation);
    fprintf('Rastrigin trigger mode: %d\n',RunSummary.BasinIsRastrigin);
    fprintf('Best Objective: %.12g\n',RunSummary.BestObjective);
    fprintf('Best FE: %d\n',RunSummary.BestFE);
    fprintf('Runtime: %.2fs\n',RunSummary.Runtime);
    fprintf('Total FE: %d\n\n',RunSummary.TotalFE);

    fprintf('RBF Evaluations: %d\n',RunSummary.RBFEvaluations);
    fprintf('RF Evaluations: %d\n',RunSummary.RFEvaluations);
    fprintf('BasinHop Evaluations: %d\n',RunSummary.BasinHopEvaluations);
    fprintf('Random Evaluations: %d\n',RunSummary.RandomEvaluations);
    fprintf('Basin trigger attempts: %d\n',RunSummary.BasinTriggerAttempts);
    fprintf('Basin triggers: %d\n',RunSummary.BasinTriggers);
    fprintf('Basin vetoes: %d\n',RunSummary.BasinVetoes);
    fprintf('Random fallbacks: %d\n',RunSummary.BasinRandomFallbacks);
    fprintf('LocalSearch Evaluations: %d\n\n',RunSummary.LocalSearchEvaluations);

    fprintf('RBF Improvements: %d\n',RunSummary.RBFImprovements);
    fprintf('RF Improvements: %d\n',RunSummary.RFImprovements);
    fprintf('BasinHop Improvements: %d\n',RunSummary.BasinHopImprovements);
    fprintf('Random Improvements: %d\n',RunSummary.RandomImprovements);
    fprintf('LocalSearch Improvements: %d\n\n',RunSummary.LocalSearchImprovements);

    fprintf('RBF MAE: %.12g, RBF RMSE: %.12g\n',RunSummary.RBF_MAE,RunSummary.RBF_RMSE);
    fprintf('RF MAE: %.12g, RF RMSE: %.12g\n',RunSummary.RF_MAE,RunSummary.RF_RMSE);
    fprintf('BasinHop MAE: %.12g, BasinHop RMSE: %.12g\n',RunSummary.BasinHop_MAE,RunSummary.BasinHop_RMSE);
    fprintf('Basin operator weights:');
    fprintf(' %.4g',RunSummary.BasinOperatorWeights);
    fprintf('\n');
    fprintf('Saved log file: %s\n',historyFile);
    fprintf('============================================\n\n');
end
