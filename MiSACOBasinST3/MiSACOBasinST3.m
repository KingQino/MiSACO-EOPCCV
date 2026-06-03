classdef MiSACOBasinST3 < ALGORITHM
% <single> <real> <expensive>
% Rastrigin-only stagnation-triggered BasinHop with conservative model-pool veto.
% Np                    --- 60   --- Size of parent population
% No                    --- 100  --- Size of offspring population
% BasinPool             --- 120  --- Number of generated basin-hop candidates
% BasinTopK             --- 30   --- Candidate shortlist size
% BasinNoveltyWeight    --- 0.20 --- Weight for mixed decision novelty
% BasinArchiveWeight    --- 0.25 --- Weight for unseen basin signatures
% DisagreementPenalty   --- 0.15 --- Penalty for tree-model disagreement
% MinDestroyRatio       --- 0.25 --- Minimum variable fraction changed by hop
% MaxDestroyRatio       --- 0.60 --- Maximum variable fraction changed by hop
% RastriginStagFE       --- 24   --- FE stagnation threshold on Rastrigin cases
% EasyTolerance         --- 1e-3 --- Disable BasinHop when best is already small
% PredImprovementTol    --- 0.02 --- Minimum conservative relative improvement
% ErrorMarginK          --- 1.00 --- Multiplier for predicted absolute error
% VetoQuantile          --- 0.60 --- Error-risk quantile kept before scoring
% MinAlign              --- 30   --- Minimum alignment samples before error model
% RBFAnomalyWeight      --- 0.10 --- Penalty for RBF/tree disagreement
% ModelCycles           --- 30   --- Tree ensemble learning cycles

    methods
        function main(Algorithm,Problem)
            [History,EvalLog,SurLog] = InitializeLogs();
            Ablation = ReadBasinAblation();
            BasinState = InitializeBasinState(Ablation);
            TriggerState = InitializeTriggerState(Problem,24,1e-3,0.02,1.00,0.60,30,0.10);
            BasinAlignDB = InitializeBasinAlignDB();
            currentBest = inf;
            runStart    = tic;
            algorithmFile = which(class(Algorithm));
            historyFile = getenv('MISACOBASINST3_HISTORY_FILE');
            if isempty(historyFile)
                historyFile = getenv('MISACOBASINST2_HISTORY_FILE');
            end
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
                historyFile = fullfile(fileparts(fileparts(algorithmFile)),'history.MiSACOBasinST3.mat');
            end

            try
                [Np,No,BasinPool,BasinTopK,BasinNoveltyWeight,BasinArchiveWeight, ...
                    DisagreementPenalty,MinDestroyRatio,MaxDestroyRatio,RastriginStagFE, ...
                    EasyTolerance,PredImprovementTol,ErrorMarginK,VetoQuantile, ...
                    MinAlign,RBFAnomalyWeight,ModelCycles] = ...
                    Algorithm.ParameterSet(60,100,120,30,0.20,0.25,0.15,0.25,0.60,24,1e-3,0.02,1.00,0.60,30,0.10,30);
                BasinPool = max(20,round(BasinPool));
                BasinTopK = max(1,round(BasinTopK));
                BasinNoveltyWeight = max(0,BasinNoveltyWeight);
                BasinArchiveWeight = max(0,BasinArchiveWeight);
                DisagreementPenalty = max(0,DisagreementPenalty);
                MinDestroyRatio = min(max(MinDestroyRatio,0.05),0.90);
                MaxDestroyRatio = min(max(MaxDestroyRatio,MinDestroyRatio),0.95);
                ErrorMarginK = max(0,ErrorMarginK);
                VetoQuantile = min(max(VetoQuantile,0.10),0.95);
                MinAlign = max(0,round(MinAlign));
                RBFAnomalyWeight = max(0,RBFAnomalyWeight);
                ModelCycles = max(5,round(ModelCycles));
                TriggerState = InitializeTriggerState( ...
                    Problem,RastriginStagFE,EasyTolerance,PredImprovementTol, ...
                    ErrorMarginK,VetoQuantile,MinAlign,RBFAnomalyWeight);

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
                    Surrogates = TrainBasinSurrogatesST3( ...
                        Train_X,Train_Y,Problem,upper_r,lower_r,ModelCycles,RBF_glo,RF_net);

                    [~,rank_v] = sort(P.objs);
                    [~,rank_v] = sort(rank_v);
                    OffDec = ACO_MV_generate(P.decs,Problem.Dr,Problem.Dc,Np,lenC,rank_v,No,upper_r,lower_r);
                    PoolPred = PredictBasinSurrogatesST3(Surrogates,OffDec,Problem,upper_r,lower_r);
                    [PoolFeatures,PoolInfo] = BuildBasinModelFeatures( ...
                        OffDec,PoolPred,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest);

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
                    BasinAlignDB = LogBasinAlignment( ...
                        BasinAlignDB,PoolFeatures(id_selected,:), ...
                        PoolInfo.EnsemblePred(id_selected),New.objs,'RBF');
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RBF',feBefore,currentBest);
                    TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];
                    [PoolFeatures,PoolInfo] = RemovePoolRow(PoolFeatures,PoolInfo,id_selected);

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
                    BasinAlignDB = LogBasinAlignment( ...
                        BasinAlignDB,PoolFeatures(id_selected,:), ...
                        PoolInfo.EnsemblePred(id_selected),New.objs,'RF');
                    [History,EvalLog,currentBest] = LogRealEvaluation(History,EvalLog,New,'RF',feBefore,currentBest);
                    TriggerState = UpdateTriggerState(TriggerState,bestBefore,currentBest,Problem.FE);
                    DB = [DB, New];
                    OffDec(id_selected,:) = [];
                    [PoolFeatures,PoolInfo] = RemovePoolRow(PoolFeatures,PoolInfo,id_selected);

                    useBasinHop = ShouldTriggerBasinHop(TriggerState,currentBest,Problem.FE);
                    if useBasinHop
                        TriggerState = RecordBasinTriggerAttempt(TriggerState);
                        [xBasin,pred_basin,opId,acceptedByVeto,feature_basin,basinInfo] = SelectBasinHopCandidate( ...
                            DB,Train_X,Train_Y,Surrogates,Problem,upper_r,lower_r,lenC, ...
                            BasinState,BasinPool,BasinTopK,BasinNoveltyWeight,BasinArchiveWeight, ...
                            DisagreementPenalty,MinDestroyRatio,MaxDestroyRatio,currentBest, ...
                            Ablation,TriggerState,BasinAlignDB);
                        TriggerState = RecordBasinSelectionInfo(TriggerState,basinInfo);
                        if acceptedByVeto
                            feBefore = Problem.FE;
                            bestBefore = currentBest;
                            New      = Problem.Evaluation(xBasin);
                            SurLog   = LogSurrogatePrediction(SurLog,'BasinHop',pred_basin,New.objs,Problem.FE);
                            BasinAlignDB = LogBasinAlignment( ...
                                BasinAlignDB,feature_basin,pred_basin,New.objs,'BasinHop');
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
                            BasinAlignDB = LogBasinAlignment( ...
                                BasinAlignDB,PoolFeatures(id_selected,:), ...
                                PoolInfo.EnsemblePred(id_selected),New.objs,'Random');
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
                        BasinAlignDB = LogBasinAlignment( ...
                            BasinAlignDB,PoolFeatures(id_selected,:), ...
                            PoolInfo.EnsemblePred(id_selected),New.objs,'Random');
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
                FinalizeLogs(History,EvalLog,SurLog,BasinState,TriggerState,BasinAlignDB,runStart,Problem,historyFile);
                rethrow(err);
            end
            FinalizeLogs(History,EvalLog,SurLog,BasinState,TriggerState,BasinAlignDB,runStart,Problem,historyFile);
        end
    end
end

function [xSelected,predSelected,opSelected,acceptedByVeto,featureSelected,info] = SelectBasinHopCandidate( ...
    DB,Train_X,Train_Y,Surrogates,Problem,upper_r,lower_r,lenC,BasinState, ...
    BasinPool,BasinTopK,BasinNoveltyWeight,BasinArchiveWeight,DisagreementPenalty, ...
    MinDestroyRatio,MaxDestroyRatio,currentBest,Ablation,TriggerState,BasinAlignDB)

    [Candidates,OpIds] = GenerateBasinHopCandidates( ...
        DB,Problem,upper_r,lower_r,lenC,BasinState,BasinPool,MinDestroyRatio,MaxDestroyRatio);
    Pred = PredictBasinSurrogatesST3(Surrogates,Candidates,Problem,upper_r,lower_r);
    [Features,FeatureInfo] = BuildBasinModelFeatures( ...
        Candidates,Pred,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest);

    if Ablation.RandomHopSelection
        idSelected = randi(size(Candidates,1));
        xSelected = Candidates(idSelected,:);
        predSelected = FeatureInfo.EnsemblePred(idSelected);
        opSelected = OpIds(idSelected);
        featureSelected = Features(idSelected,:);
        info = DefaultBasinSelectionInfo();
        info.Score = nan;
        info.Quality = FeatureInfo.Quality(idSelected);
        info.Novelty = FeatureInfo.MixedNovelty(idSelected);
        acceptedByVeto = true;
        return;
    end

    ensemblePred = FeatureInfo.EnsemblePred;
    quality = FeatureInfo.Quality;
    mixedNovelty = FeatureInfo.MixedNovelty;
    if Ablation.UseArchive
        basinNovelty = BasinArchiveNovelty(Candidates,Train_X,BasinState,Problem,upper_r,lower_r);
    else
        basinNovelty = zeros(size(Candidates,1),1);
        BasinArchiveWeight = 0;
    end
    disagreement = FeatureInfo.Disagreement;

    qualityIdx = ShortlistByPrediction(FeatureInfo.TreePred,BasinTopK);
    noveltyIdx = TopKIndices(mixedNovelty + basinNovelty,max(1,ceil(BasinTopK/2)));
    candidateIdx = unique([qualityIdx(:); noveltyIdx(:)]);
    if isempty(candidateIdx)
        candidateIdx = (1:size(Candidates,1))';
    end

    [predAbsError,predLogError,usedAlignment] = PredictBasinAlignmentError( ...
        BasinAlignDB,Features,TriggerState.MinAlign,Train_Y,currentBest);
    conservativePred = ensemblePred + TriggerState.ErrorMarginK*predAbsError;
    errorRisk = UnitNormalize(predAbsError);
    activeIdx = candidateIdx;
    vetoed = false(size(candidateIdx));
    if usedAlignment && numel(candidateIdx) > 1
        shortlistRisk = errorRisk(candidateIdx);
        vetoThreshold = QuantileFinite(shortlistRisk,TriggerState.VetoQuantile);
        vetoed = isfinite(shortlistRisk) & shortlistRisk >= vetoThreshold;
        activeIdx = candidateIdx(~vetoed);
        if isempty(activeIdx)
            activeIdx = candidateIdx;
            vetoed = false(size(candidateIdx));
        end
    end

    score = quality + ...
        BasinNoveltyWeight*mixedNovelty + ...
        BasinArchiveWeight*basinNovelty + ...
        0.10*FeatureInfo.ModelCount - ...
        DisagreementPenalty*disagreement - ...
        TriggerState.RBFAnomalyWeight*FeatureInfo.RBFAnomaly - ...
        0.20*errorRisk;
    score(~isfinite(score)) = -inf;
    if all(isinf(score(activeIdx)))
        score = mixedNovelty + basinNovelty;
        score(~isfinite(score)) = 0;
    end

    [~,localBest] = max(score(activeIdx));
    idSelected = activeIdx(localBest);

    xSelected = Candidates(idSelected,:);
    predSelected = ensemblePred(idSelected);
    opSelected = OpIds(idSelected);
    featureSelected = Features(idSelected,:);
    acceptedByVeto = PassConservativeVeto( ...
        conservativePred(idSelected),currentBest,TriggerState.PredImprovementTol, ...
        usedAlignment,FeatureInfo.ModelCount(idSelected));

    info = DefaultBasinSelectionInfo();
    info.UsedAlignment = usedAlignment;
    info.PredLogError = predLogError(idSelected);
    info.PredAbsError = predAbsError(idSelected);
    info.ConservativePred = conservativePred(idSelected);
    info.Score = score(idSelected);
    info.Quality = quality(idSelected);
    info.Novelty = mixedNovelty(idSelected);
    info.Disagreement = disagreement(idSelected);
    info.RBFAnomaly = FeatureInfo.RBFAnomaly(idSelected);
    info.VetoedCandidates = sum(vetoed);
end

function accepted = PassConservativeVeto(conservativePred,currentBest,predImprovementTol,usedAlignment,modelCount)
    if ~usedAlignment || ~isfinite(conservativePred) || ~isfinite(currentBest) || modelCount < 0.50
        accepted = false;
        return;
    end

    predictedImprovement = currentBest - conservativePred;
    relativeImprovement = predictedImprovement/max(1,abs(currentBest));
    accepted = predictedImprovement > 0 && relativeImprovement >= predImprovementTol;
end

function Surrogates = TrainBasinSurrogatesST3(Train_X,Train_Y,Problem,upper_r,lower_r,ModelCycles,RBF_glo,RF_net)
    catIdx = Problem.Dr+1:Problem.D;
    y = Train_Y(:,1);
    treeMinLeaf = max(1,ceil(size(Train_X,1)*0.02));

    Surrogates.Names = {'RBF','RF','ET','GBDT'};
    Surrogates.RBF = RBF_glo;
    Surrogates.RF = RF_net;
    Surrogates.ET = [];
    Surrogates.GBDT = [];

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

    if isempty(Surrogates.RBF)
        try
            Surrogates.RBF = RBFCreate_Hamming( ...
                Train_X,y,Problem.Dc,upper_r,lower_r,'gaussian');
        catch
            Surrogates.RBF = [];
        end
    end
end

function Pred = PredictBasinSurrogatesST3(Surrogates,X,Problem,upper_r,lower_r)
    n = size(X,1);
    Pred = nan(n,numel(Surrogates.Names));

    if ~isempty(Surrogates.RBF)
        Pred(:,1) = PredictRBF(X,Surrogates.RBF,Problem,upper_r,lower_r);
    end
    if ~isempty(Surrogates.RF)
        Pred(:,2) = SafePredict(Surrogates.RF,X);
    end
    if ~isempty(Surrogates.ET)
        Pred(:,3) = SafePredict(Surrogates.ET,X);
    end
    if ~isempty(Surrogates.GBDT)
        Pred(:,4) = SafePredict(Surrogates.GBDT,X);
    end
end

function [Features,Info] = BuildBasinModelFeatures( ...
    X,Pred,Train_X,Train_Y,Problem,upper_r,lower_r,currentBest)

    n = size(X,1);
    Dr = Problem.Dr;
    Dc = Problem.Dc;

    xrange = upper_r - lower_r;
    xrange(xrange < eps) = eps;
    x_r = zeros(n,0);
    if Dr > 0
        x_r = (X(:,1:Dr) - lower_r)./xrange;
        x_r = min(max(x_r,0),1);
    end

    x_c_scaled = zeros(n,0);
    if Dc > 0
        x_c = ceil(X(:,Dr+1:Problem.D));
        len_c = NormalizeLenC(Problem.len_c,Dc);
        crange = max(1,len_c - 1);
        x_c_scaled = (x_c - 1)./crange;
        x_c_scaled = min(max(x_c_scaled,0),1);
    end

    predRank = nan(size(Pred));
    for j = 1:size(Pred,2)
        predRank(:,j) = NormalizedRank(Pred(:,j));
    end

    treePreds = Pred(:,2:end);
    treeRanks = predRank(:,2:end);
    treePred = RowMedianFinite(treePreds);
    fallbackPred = RowMedianFinite(Pred);
    missingTree = ~isfinite(treePred);
    treePred(missingTree) = fallbackPred(missingTree);
    ensemblePred = treePred;
    predMean = RowMeanFinite(treePreds);
    missingMean = ~isfinite(predMean);
    predMean(missingMean) = RowMeanFinite(Pred(missingMean,:));
    predStd = RowStdFinite(treePreds);
    voteRank = RowMeanFinite(treeRanks);
    missingVote = ~isfinite(voteRank);
    voteRank(missingVote) = RowMeanFinite(predRank(missingVote,:));
    modelCount = sum(isfinite(Pred),2)./size(Pred,2);

    [nearestDist,avgKDist,sameCatCount] = CandidateDistances(X,Train_X,Problem,upper_r,lower_r);
    distanceNovelty = UnitNormalize(nearestDist);
    ensembleQuality = 1 - NormalizedRank(treePred);
    voteQuality = 1 - voteRank;
    quality = 0.65*ensembleQuality + 0.35*voteQuality;
    quality(~isfinite(quality)) = 0;

    bestNorm = NormalizeWithReference(currentBest,Train_Y);
    sameCatSeen = sameCatCount > 0;
    sameCatScaled = min(1,sameCatCount./max(1,size(Train_X,1)));
    categoryNovelty = 1 - sameCatScaled;
    categoryWeight = Problem.Dc./max(1,Problem.D);
    mixedNovelty = (1 - categoryWeight).*distanceNovelty + categoryWeight.*categoryNovelty;
    disagreement = UnitNormalize(RowStdFinite(treeRanks));
    rbfAnomalyRaw = abs(Pred(:,1) - treePred);
    rbfAnomalyRaw(~isfinite(rbfAnomalyRaw)) = 0;
    rbfAnomaly = UnitNormalize(rbfAnomalyRaw);
    predImprovement = currentBest - ensemblePred;

    Features = [ ...
        x_r, ...
        x_c_scaled, ...
        UnitNormalizeColumns(Pred), ...
        predRank, ...
        UnitNormalize(ensemblePred), ...
        UnitNormalize(predMean), ...
        UnitNormalize(predStd), ...
        disagreement, ...
        rbfAnomaly, ...
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
    Info.MixedNovelty = mixedNovelty;
    Info.Disagreement = disagreement;
    Info.EnsemblePred = ensemblePred;
    Info.TreePred = treePred;
    Info.RBFAnomaly = rbfAnomaly;
    Info.ModelCount = modelCount;
end

function [PoolFeatures,PoolInfo] = RemovePoolRow(PoolFeatures,PoolInfo,id)
    PoolFeatures(id,:) = [];
    fields = fieldnames(PoolInfo);
    for i = 1:numel(fields)
        value = PoolInfo.(fields{i});
        if isnumeric(value) && size(value,1) >= id
            value(id,:) = [];
            PoolInfo.(fields{i}) = value;
        end
    end
end

function [predAbsError,predLogError,usedAlignment] = PredictBasinAlignmentError( ...
    BasinAlignDB,Features,MinAlign,Train_Y,currentBest)

    n = size(Features,1);
    usedAlignment = false;
    baseline = BasinAlignmentErrorBaseline(BasinAlignDB,Train_Y,currentBest);
    predAbsError = repmat(baseline,n,1);
    predLogError = log(1 + predAbsError);

    if ~HasBasinAlignmentData(BasinAlignDB,MinAlign)
        return;
    end

    y = BasinAlignDB.LogAbsError(:);
    X = BasinAlignDB.Features;
    valid = all(isfinite(X),2) & isfinite(y);
    X = X(valid,:);
    y = y(valid,:);
    if size(X,1) < max(2,MinAlign)
        return;
    end

    if max(y) - min(y) < eps
        predLogError(:) = mean(y);
        predAbsError = exp(predLogError) - 1;
        usedAlignment = true;
        return;
    end

    try
        model = fitrensemble(X,y,'Method','Bag','NumLearningCycles',30,'Learners','tree');
        predLogError = predict(model,Features);
        predLogError = max(predLogError,0);
        if all(isfinite(predLogError))
            predAbsError = exp(predLogError) - 1;
            usedAlignment = true;
        else
            predAbsError(:) = baseline;
            predLogError(:) = log(1 + baseline);
        end
    catch
        predAbsError(:) = baseline;
        predLogError(:) = log(1 + baseline);
    end
end

function baseline = BasinAlignmentErrorBaseline(BasinAlignDB,Train_Y,currentBest)
    if ~isempty(BasinAlignDB.LogAbsError)
        absError = exp(BasinAlignDB.LogAbsError(:)) - 1;
        absError = absError(isfinite(absError));
        if ~isempty(absError)
            baseline = QuantileFinite(absError,0.75);
            return;
        end
    end

    ref = [Train_Y(:); currentBest];
    ref = ref(isfinite(ref));
    if isempty(ref)
        baseline = inf;
    else
        baseline = max(1,0.25*(max(ref) - min(ref)));
    end
end

function tf = HasBasinAlignmentData(BasinAlignDB,MinAlign)
    tf = size(BasinAlignDB.Features,1) >= MinAlign && ...
         numel(unique(round(BasinAlignDB.LogAbsError*1e8)/1e8)) >= 2;
end

function AlignDB = InitializeBasinAlignDB()
    AlignDB.Features = [];
    AlignDB.LogAbsError = [];
    AlignDB.Source = {};
end

function AlignDB = LogBasinAlignment(AlignDB,feature,predicted,realValue,source)
    predicted = predicted(1);
    realValue = realValue(1);
    absError = abs(predicted - realValue);
    if isfinite(absError) && all(isfinite(feature))
        AlignDB.Features(end+1,:) = feature;
        AlignDB.LogAbsError(end+1,1) = log(1 + absError);
        AlignDB.Source{end+1,1} = source;
    end

    maxSamples = 300;
    if size(AlignDB.Features,1) > maxSamples
        AlignDB.Features = AlignDB.Features(end-maxSamples+1:end,:);
        AlignDB.LogAbsError = AlignDB.LogAbsError(end-maxSamples+1:end,:);
        AlignDB.Source = AlignDB.Source(end-maxSamples+1:end,:);
    end
end

function info = DefaultBasinSelectionInfo()
    info.UsedAlignment = false;
    info.PredLogError = nan;
    info.PredAbsError = nan;
    info.ConservativePred = nan;
    info.Score = nan;
    info.Quality = nan;
    info.Novelty = nan;
    info.Disagreement = nan;
    info.RBFAnomaly = nan;
    info.VetoedCandidates = 0;
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
        if iscell(pred)
            pred = str2double(pred);
        end
        pred = pred(:);
        if numel(pred) ~= size(X,1)
            pred = nan(size(X,1),1);
        end
        pred(~isfinite(pred)) = nan;
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

function [nearestDist,avgKDist,sameCatCount] = CandidateDistances(X,Train_X,Problem,upper_r,lower_r)
    n  = size(X,1);
    nt = size(Train_X,1);
    Dr = Problem.Dr;
    Dc = Problem.Dc;

    nearestDist = zeros(n,1);
    avgKDist = zeros(n,1);
    sameCatCount = zeros(n,1);
    if nt == 0
        nearestDist(:) = 1;
        avgKDist(:) = 1;
        return;
    end

    scale = max(upper_r - lower_r,eps);
    trainCat = ceil(Train_X(:,Dr+1:Dr+Dc));
    for i = 1:n
        dist = zeros(nt,1);
        parts = 0;
        if Dr > 0
            diffR = (Train_X(:,1:Dr) - X(i,1:Dr))./scale;
            dist = dist + sqrt(mean(diffR.^2,2));
            parts = parts + 1;
        end
        if Dc > 0
            candCat = ceil(X(i,Dr+1:Dr+Dc));
            diffC = mean(trainCat ~= candCat,2);
            dist = dist + diffC;
            parts = parts + 1;
            sameCatCount(i) = sum(diffC == 0);
        end
        if parts > 0
            dist = dist./parts;
        end
        dist = sort(dist,'ascend');
        nearestDist(i) = dist(1);
        k = min(5,numel(dist));
        avgKDist(i) = mean(dist(1:k));
    end
end

function ranks = NormalizedRank(values)
    values = values(:);
    ranks = nan(size(values));
    finite = isfinite(values);
    if ~any(finite)
        return;
    end
    idx = find(finite);
    [~,order] = sort(values(finite),'ascend');
    rankVals = zeros(numel(order),1);
    if numel(order) == 1
        rankVals(order) = 0;
    else
        rankVals(order) = (0:numel(order)-1)'./(numel(order)-1);
    end
    ranks(idx) = rankVals;
end

function out = UnitNormalizeColumns(values)
    out = zeros(size(values));
    for j = 1:size(values,2)
        out(:,j) = UnitNormalize(values(:,j));
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

function out = RowMeanFinite(values)
    out = nan(size(values,1),1);
    for i = 1:size(values,1)
        row = values(i,:);
        row = row(isfinite(row));
        if ~isempty(row)
            out(i) = mean(row);
        end
    end
end

function out = RowStdFinite(values)
    out = nan(size(values,1),1);
    for i = 1:size(values,1)
        row = values(i,:);
        row = row(isfinite(row));
        if numel(row) == 1
            out(i) = 0;
        elseif numel(row) > 1
            out(i) = std(row);
        end
    end
end

function value = NormalizeWithReference(value,reference)
    ref = [reference(:); value];
    ref = ref(isfinite(ref));
    if isempty(ref)
        value = 0.5;
        return;
    end
    low = min(ref);
    high = max(ref);
    if high <= low || ~isfinite(value)
        value = 0.5;
    else
        value = (value - low)./(high - low);
        value = min(max(value,0),1);
    end
end

function out = ReplaceNonFinite(values,replacement)
    out = values;
    out(~isfinite(out)) = replacement;
end

function q = QuantileFinite(values,probability)
    values = values(:);
    values = sort(values(isfinite(values)),'ascend');
    if isempty(values)
        q = nan;
        return;
    end
    probability = min(max(probability,0),1);
    idx = max(1,min(numel(values),ceil(probability*numel(values))));
    q = values(idx);
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

function TriggerState = InitializeTriggerState( ...
    Problem,RastriginStagFE,EasyTolerance,PredImprovementTol, ...
    ErrorMarginK,VetoQuantile,MinAlign,RBFAnomalyWeight)
    TriggerState.IsRastrigin = IsRastriginProblem(Problem);
    TriggerState.LastImprovementFE = Problem.FE;
    TriggerState.LastBasinTriggerFE = -inf;
    TriggerState.BasinTriggerAttempts = 0;
    TriggerState.BasinTriggers = 0;
    TriggerState.BasinVetoes = 0;
    TriggerState.BasinAlignmentGuidedAttempts = 0;
    TriggerState.BasinCandidateVetoes = 0;
    TriggerState.RandomFallbacks = 0;
    TriggerState.RastriginStagFE = max(1,round(RastriginStagFE));
    TriggerState.RastriginCooldownFE = max(1,round(RastriginStagFE));
    TriggerState.EasyTolerance = max(0,EasyTolerance);
    TriggerState.PredImprovementTol = max(0,PredImprovementTol);
    TriggerState.ErrorMarginK = max(0,ErrorMarginK);
    TriggerState.VetoQuantile = min(max(VetoQuantile,0.10),0.95);
    TriggerState.MinAlign = max(0,round(MinAlign));
    TriggerState.RBFAnomalyWeight = max(0,RBFAnomalyWeight);
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

function TriggerState = RecordBasinSelectionInfo(TriggerState,info)
    if info.UsedAlignment
        TriggerState.BasinAlignmentGuidedAttempts = TriggerState.BasinAlignmentGuidedAttempts + 1;
    end
    TriggerState.BasinCandidateVetoes = TriggerState.BasinCandidateVetoes + info.VetoedCandidates;
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
    rawMode = strtrim(getenv('MISACOBASINST3_ABLATION'));
    if isempty(rawMode)
        rawMode = strtrim(getenv('MISACOBASINST2_ABLATION'));
    end
    if isempty(rawMode)
        rawMode = strtrim(getenv('MISACOBASINST_ABLATION'));
    end
    if isempty(rawMode)
        rawMode = 'conservative_veto';
    end

    mode = lower(strrep(rawMode,'-','_'));
    switch mode
        case {'conservative_veto','st3','rastrigin_veto','stagnation_triggered','st','full','none','default'}
            Ablation.Name = 'conservative_veto';
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
            error('MiSACOBasinST3:InvalidAblation', ...
                ['Unknown MISACOBASINST3_ABLATION=%s. Valid values are ', ...
                'conservative_veto, no_archive, fixed_op, random_hop.'], rawMode);
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

function RunSummary = FinalizeLogs(History,EvalLog,SurLog,BasinState,TriggerState,BasinAlignDB,runStart,Problem,historyFile)
    RunSummary = BuildRunSummary(History,EvalLog,SurLog,BasinState,TriggerState,BasinAlignDB,runStart,Problem);
    save(historyFile,'History','EvalLog','SurLog','BasinAlignDB','RunSummary');
    PrintRunSummary(RunSummary,historyFile);
end

function RunSummary = BuildRunSummary(~,EvalLog,SurLog,BasinState,TriggerState,BasinAlignDB,runStart,Problem)
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
    RunSummary.BasinAlignmentSamples = size(BasinAlignDB.Features,1);
    RunSummary.BasinAlignmentGuidedAttempts = TriggerState.BasinAlignmentGuidedAttempts;
    RunSummary.BasinCandidateVetoes = TriggerState.BasinCandidateVetoes;
    RunSummary.BasinRandomFallbacks = TriggerState.RandomFallbacks;
    RunSummary.BasinLastImprovementFE = TriggerState.LastImprovementFE;
    RunSummary.BasinRastriginStagFE = TriggerState.RastriginStagFE;
    RunSummary.BasinEasyTolerance = TriggerState.EasyTolerance;
    RunSummary.BasinPredImprovementTol = TriggerState.PredImprovementTol;
    RunSummary.BasinErrorMarginK = TriggerState.ErrorMarginK;
    RunSummary.BasinVetoQuantile = TriggerState.VetoQuantile;
    RunSummary.BasinMinAlign = TriggerState.MinAlign;
    RunSummary.BasinRBFAnomalyWeight = TriggerState.RBFAnomalyWeight;

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
    fprintf('\n========== MiSACOBasinST3 Run Summary ==========\n');
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
    fprintf('Basin alignment samples: %d\n',RunSummary.BasinAlignmentSamples);
    fprintf('Basin alignment-guided attempts: %d\n',RunSummary.BasinAlignmentGuidedAttempts);
    fprintf('Basin candidate vetoes: %d\n',RunSummary.BasinCandidateVetoes);
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
