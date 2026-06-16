function instanceSummary = RunEOPCCVExperiment(problemId, seeds, maxFE, parentN, offspringN, statsRoot, numWorkers, algorithmName)
%RunEOPCCVExperiment Run one EOPCCV problem for multiple random seeds.

    if nargin < 2 || isempty(seeds)
        seeds = 1:10;
    end
    if nargin < 3 || isempty(maxFE)
        maxFE = 600;
    end
    if nargin < 4 || isempty(parentN)
        parentN = 60;
    end
    if nargin < 5 || isempty(offspringN)
        offspringN = 100;
    end
    if nargin < 6 || isempty(statsRoot)
        statsRoot = fullfile(fileparts(mfilename('fullpath')),'MiSACO','stats');
    end
    if nargin < 7 || isempty(numWorkers)
        numWorkers = ReadIntegerEnv('MISACO_NUM_WORKERS', 1);
    end
    if nargin < 8 || isempty(algorithmName)
        algorithmName = ReadStringEnv('MISACO_ALGORITHM', 'MiSACO');
    end

    problemNumber = ParseProblemId(problemId);
    instanceName  = sprintf('F%d',problemNumber);
    problemClass  = sprintf('EOPCCV_F%d',problemNumber);
    problemHandle = str2func(problemClass);

    if isempty(which(problemClass))
        error('RunEOPCCVExperiment:ProblemNotFound', ...
            'Problem class is not on the MATLAB path: %s', problemClass);
    end

    instanceDir = fullfile(statsRoot,instanceName);
    EnsureFolder(instanceDir);

    saveDetailedLogs = ReadBooleanEnv('MISACO_SAVE_DETAILED_LOGS', false);
    writeInstanceStats = ReadBooleanEnv('MISACO_WRITE_INSTANCE_STATS', true);
    numWorkers = max(1,min(numWorkers,numel(seeds)));
    useParallel = numWorkers > 1;
    if useParallel && ~HasParallelSupport()
        warning('RunEOPCCVExperiment:NoParallelSupport', ...
            ['Parallel Computing Toolbox support unavailable. ' ...
            'Falling back to serial seed execution.']);
        useParallel = false;
        numWorkers = 1;
    end

    fprintf('Running %s for seeds %s\n', problemClass, mat2str(seeds));
    fprintf('Algorithm: %s\n', algorithmName);
    fprintf('Seed workers: %d\n', numWorkers);
    fprintf('Detailed CSV/result logs: %d\n', saveDetailedLogs);
    fprintf('Write instance stats now: %d\n', writeInstanceStats);
    seedSummaries = repmat(EmptySeedSummary(), numel(seeds), 1);

    if useParallel
        pool = EnsureLocalPool(numWorkers);
        fprintf('Running seeds in parallel with %d worker(s).\n', pool.NumWorkers);
        parfor k = 1:numel(seeds)
            seed = seeds(k);
            seedDir = fullfile(instanceDir,num2str(seed));
            EnsureFolder(seedDir);
            seedSummaries(k) = RunSingleSeed(problemHandle, problemClass, ...
                instanceName, seed, maxFE, parentN, offspringN, seedDir, saveDetailedLogs, algorithmName);
        end
    else
        fprintf('Running seeds serially.\n');
        for k = 1:numel(seeds)
            seed = seeds(k);
            seedDir = fullfile(instanceDir,num2str(seed));
            EnsureFolder(seedDir);
            seedSummaries(k) = RunSingleSeed(problemHandle, problemClass, ...
                instanceName, seed, maxFE, parentN, offspringN, seedDir, saveDetailedLogs, algorithmName);
        end
    end

    if writeInstanceStats
        instanceSummary = WriteInstanceStats(instanceDir, instanceName, seedSummaries);
        fprintf('Completed %s | best=%.12g mean=%.12g std=%.12g completed=%d\n', ...
            instanceName, instanceSummary.MinBestObjective, ...
            instanceSummary.MeanBestObjective, instanceSummary.StdBestObjective, ...
            instanceSummary.CompletedSeeds);
    else
        instanceSummary = struct( ...
            'Instance', instanceName, ...
            'CompletedSeeds', numel(seedSummaries), ...
            'MinBestObjective', min([seedSummaries.BestObjective]), ...
            'MeanBestObjective', mean([seedSummaries.BestObjective]), ...
            'StdBestObjective', std([seedSummaries.BestObjective]), ...
            'MaxBestObjective', max([seedSummaries.BestObjective]), ...
            'MeanRuntime', mean([seedSummaries.Runtime]), ...
            'MeanTotalFE', mean([seedSummaries.TotalFE]));
        fprintf('Completed %s seed task(s); instance stats deferred to collect job.\n', ...
            instanceName);
    end
end


function summary = RunSingleSeed(problemHandle, problemClass, instanceName, seed, maxFE, parentN, offspringN, seedDir, saveDetailedLogs, algorithmName)
    historyFile = fullfile(seedDir, sprintf('history.%s.seed%d.mat', instanceName, seed));
    oldHistoryFile = getenv('MISACO_HISTORY_FILE');
    setenv('MISACO_HISTORY_FILE',historyFile);
    cleanup = onCleanup(@() setenv('MISACO_HISTORY_FILE',oldHistoryFile));

    rng(seed,'twister');
    fprintf('  Seed %d -> %s\n', seed, seedDir);

    Problem   = problemHandle('maxFE',maxFE);
    AlgorithmHandle = str2func(algorithmName);
    Algorithm = AlgorithmHandle('parameter',{parentN,offspringN},'save',0);
    Algorithm.Solve(Problem);

    Population = Algorithm.result{end,2};
    popDecs = Population.decs;
    popObjs = Population.objs;
    [bestObj,bestId] = min(popObjs);
    bestX = popDecs(bestId,:);

    if ~isfile(historyFile)
        error('RunEOPCCVExperiment:MissingHistoryFile', ...
            '%s did not write the expected history file: %s', algorithmName, historyFile);
    end

    data = load(historyFile);
    History = data.History;
    EvalLog = data.EvalLog;
    SurLog  = data.SurLog;
    RunSummary = data.RunSummary;
    hasAlignLog = isfield(data,'AlignLog');
    if hasAlignLog
        AlignLog = data.AlignLog;
    end
    hasBasinAlignDB = isfield(data,'BasinAlignDB');
    if hasBasinAlignDB
        BasinAlignDB = data.BasinAlignDB;
    end
    RunSummary.Algorithm    = algorithmName;
    RunSummary.Problem      = problemClass;
    RunSummary.Instance     = instanceName;
    RunSummary.Seed         = seed;
    RunSummary.MaxFE        = maxFE;
    RunSummary.ParentN      = parentN;
    RunSummary.OffspringN   = offspringN;
    RunSummary.BestDecision = bestX;
    RunSummary.SaveDetailedLogs = saveDetailedLogs;
    saveVars = {'History','EvalLog','SurLog','RunSummary'};
    if hasAlignLog
        saveVars{end+1} = 'AlignLog'; %#ok<AGROW>
    end
    if hasBasinAlignDB
        saveVars{end+1} = 'BasinAlignDB'; %#ok<AGROW>
    end
    save(historyFile,saveVars{:});

    WriteHistoryCsv(fullfile(seedDir,sprintf('evols.%s.csv',instanceName)),History);
    WriteSolutionFile(fullfile(seedDir,sprintf('solution.%s.txt',instanceName)), ...
        bestObj, RunSummary.BestFE, bestX);
    WriteSeedSummaryFile(fullfile(seedDir,sprintf('summary.%s.seed%d.txt',instanceName,seed)),RunSummary);

    if saveDetailedLogs
        WriteEvalLogCsv(fullfile(seedDir,sprintf('eval_log.%s.csv',instanceName)),EvalLog);
        WriteSurLogCsv(fullfile(seedDir,sprintf('surrogate_log.%s.csv',instanceName)),SurLog);
        if hasAlignLog
            WriteAlignLogCsv(fullfile(seedDir,sprintf('alignment_log.%s.csv',instanceName)),AlignLog);
        end

        result = struct();
        result.RunSummary = RunSummary;
        if hasAlignLog
            result.AlignLog = AlignLog;
        end
        result.bestObjective = bestObj;
        result.bestFE = RunSummary.BestFE;
        result.bestDecision = bestX;
        result.finalPopulationDecs = popDecs;
        result.finalPopulationObjs = popObjs;
        save(fullfile(seedDir,sprintf('result.%s.seed%d.mat',instanceName,seed)),'result');
    end

    summary = EmptySeedSummary();
    summary.Seed = seed;
    summary.BestObjective = bestObj;
    summary.BestFE = RunSummary.BestFE;
    summary.Runtime = RunSummary.Runtime;
    summary.TotalFE = RunSummary.TotalFE;
    summary.RBFEvaluations = RunSummary.RBFEvaluations;
    summary.RFEvaluations = RunSummary.RFEvaluations;
    summary.RandomEvaluations = RunSummary.RandomEvaluations;
    summary.BasinHopEvaluations = GetSummaryField(RunSummary,'BasinHopEvaluations');
    summary.BasinHopExploreEvaluations = GetSummaryField(RunSummary,'BasinHopExploreEvaluations');
    summary.LocalSearchEvaluations = RunSummary.LocalSearchEvaluations;
    summary.LocalSearchBestOnlyEvaluations = GetSummaryField(RunSummary,'LocalSearchBestOnlyEvaluations');
    summary.LocalSearchMultiStartEvaluations = GetSummaryField(RunSummary,'LocalSearchMultiStartEvaluations');
    summary.LocalSearchCategoryEvaluations = GetSummaryField(RunSummary,'LocalSearchCategoryEvaluations');
    summary.RBFImprovements = RunSummary.RBFImprovements;
    summary.RFImprovements = RunSummary.RFImprovements;
    summary.RandomImprovements = RunSummary.RandomImprovements;
    summary.BasinHopImprovements = GetSummaryField(RunSummary,'BasinHopImprovements');
    summary.BasinHopExploreImprovements = GetSummaryField(RunSummary,'BasinHopExploreImprovements');
    summary.LocalSearchImprovements = RunSummary.LocalSearchImprovements;
    summary.LocalSearchBestOnlyImprovements = GetSummaryField(RunSummary,'LocalSearchBestOnlyImprovements');
    summary.LocalSearchMultiStartImprovements = GetSummaryField(RunSummary,'LocalSearchMultiStartImprovements');
    summary.LocalSearchCategoryImprovements = GetSummaryField(RunSummary,'LocalSearchCategoryImprovements');
    summary.RBF_MAE = RunSummary.RBF_MAE;
    summary.RBF_RMSE = RunSummary.RBF_RMSE;
    summary.RF_MAE = RunSummary.RF_MAE;
    summary.RF_RMSE = RunSummary.RF_RMSE;
    summary.BasinHop_MAE = GetSummaryField(RunSummary,'BasinHop_MAE');
    summary.BasinHop_RMSE = GetSummaryField(RunSummary,'BasinHop_RMSE');
    summary.BasinHopExplore_MAE = GetSummaryField(RunSummary,'BasinHopExplore_MAE');
    summary.BasinHopExplore_RMSE = GetSummaryField(RunSummary,'BasinHopExplore_RMSE');
    summary.BasinTriggerAttempts = GetSummaryField(RunSummary,'BasinTriggerAttempts');
    summary.BasinTriggers = GetSummaryField(RunSummary,'BasinTriggers');
    summary.BasinVetoes = GetSummaryField(RunSummary,'BasinVetoes');
    summary.BasinExploreTriggers = GetSummaryField(RunSummary,'BasinExploreTriggers');
    summary.BasinAlignmentSamples = GetSummaryField(RunSummary,'BasinAlignmentSamples');
    summary.BasinAlignmentGuidedAttempts = GetSummaryField(RunSummary,'BasinAlignmentGuidedAttempts');
    summary.BasinCandidateVetoes = GetSummaryField(RunSummary,'BasinCandidateVetoes');
    summary.BasinRandomFallbacks = GetSummaryField(RunSummary,'BasinRandomFallbacks');
    summary.BasinIsRastrigin = GetSummaryField(RunSummary,'BasinIsRastrigin');
    summary.EnsembleEvaluations = GetSummaryField(RunSummary,'EnsembleEvaluations');
    summary.EnsembleImprovements = GetSummaryField(RunSummary,'EnsembleImprovements');
    summary.ET_MAE = GetSummaryField(RunSummary,'ET_MAE');
    summary.ET_RMSE = GetSummaryField(RunSummary,'ET_RMSE');
    summary.GBDT_MAE = GetSummaryField(RunSummary,'GBDT_MAE');
    summary.GBDT_RMSE = GetSummaryField(RunSummary,'GBDT_RMSE');
    summary.Ensemble_MAE = GetSummaryField(RunSummary,'Ensemble_MAE');
    summary.Ensemble_RMSE = GetSummaryField(RunSummary,'Ensemble_RMSE');
    summary.AlignmentSamples = GetSummaryField(RunSummary,'AlignmentSamples');
    summary.AlignmentGuidedEvaluations = GetSummaryField(RunSummary,'AlignmentGuidedEvaluations');
    summary.AlignmentFallbackEvaluations = GetSummaryField(RunSummary,'AlignmentFallbackEvaluations');
    summary.AlignmentError_MAE = GetSummaryField(RunSummary,'AlignmentError_MAE');
    summary.AlignmentError_RMSE = GetSummaryField(RunSummary,'AlignmentError_RMSE');
    summary.MSLocalStartAttempts = GetSummaryField(RunSummary,'MSLocalStartAttempts');
    summary.MSLocalSupportedCategories = GetSummaryField(RunSummary,'MSLocalSupportedCategories');
    summary.MSLocalCategoryStarts = GetSummaryField(RunSummary,'MSLocalCategoryStarts');
    summary.MSLocalDBStarts = GetSummaryField(RunSummary,'MSLocalDBStarts');
    summary.MSLocalACOStarts = GetSummaryField(RunSummary,'MSLocalACOStarts');
    summary.MSLocalSurrogateAttempts = GetSummaryField(RunSummary,'MSLocalSurrogateAttempts');
    summary.MSLocalCandidates = GetSummaryField(RunSummary,'MSLocalCandidates');
    summary.MSLocalBestCandidates = GetSummaryField(RunSummary,'MSLocalBestCandidates');
    summary.MSLocalMultiCandidates = GetSummaryField(RunSummary,'MSLocalMultiCandidates');
    summary.MSLocalCategoryCandidates = GetSummaryField(RunSummary,'MSLocalCategoryCandidates');
    summary.MSLocalDBCandidates = GetSummaryField(RunSummary,'MSLocalDBCandidates');
    summary.MSLocalACOCandidates = GetSummaryField(RunSummary,'MSLocalACOCandidates');
    summary.MSLocalGatedCandidates = GetSummaryField(RunSummary,'MSLocalGatedCandidates');
    summary.MSLocalSuccesses = GetSummaryField(RunSummary,'MSLocalSuccesses');
    summary.MSLocalSwitches = GetSummaryField(RunSummary,'MSLocalSwitches');
    summary.MSLocalFallbacks = GetSummaryField(RunSummary,'MSLocalFallbacks');
    summary.MSLocalSkippedByGate = GetSummaryField(RunSummary,'MSLocalSkippedByGate');
    summary.MSLocalMeanSelectedPred = GetSummaryField(RunSummary,'MSLocalMeanSelectedPred');
    summary.MSLocalMeanSelectedPredStart = GetSummaryField(RunSummary,'MSLocalMeanSelectedPredStart');
    summary.MSLocalMeanSelectedPredImprovement = GetSummaryField(RunSummary,'MSLocalMeanSelectedPredImprovement');
    summary.MSLocalMeanSelectedNovelty = GetSummaryField(RunSummary,'MSLocalMeanSelectedNovelty');
    summary.MSLocalMeanSelectedStartObjective = GetSummaryField(RunSummary,'MSLocalMeanSelectedStartObjective');
    summary.MSLocalBestStartSelectionRate = GetSummaryField(RunSummary,'MSLocalBestStartSelectionRate');
    summary.MSLocalCategorySelectionRate = GetSummaryField(RunSummary,'MSLocalCategorySelectionRate');
    summary.MSLocalMeanSelectedCategorySamples = GetSummaryField(RunSummary,'MSLocalMeanSelectedCategorySamples');
    summary.MSLocalACOStartSelectionRate = GetSummaryField(RunSummary,'MSLocalACOStartSelectionRate');
    summary.MSLocalAdaptiveBestOnlyModes = GetSummaryField(RunSummary,'MSLocalAdaptiveBestOnlyModes');
    summary.MSLocalAdaptiveMultiModes = GetSummaryField(RunSummary,'MSLocalAdaptiveMultiModes');
    summary.MSLocalAdaptiveStagnationTriggers = GetSummaryField(RunSummary,'MSLocalAdaptiveStagnationTriggers');
    summary.MSLocalAdaptiveGoodRateTriggers = GetSummaryField(RunSummary,'MSLocalAdaptiveGoodRateTriggers');
    summary.MSLocalAdaptiveLowRateBlocks = GetSummaryField(RunSummary,'MSLocalAdaptiveLowRateBlocks');
    summary.MSLocalAdaptiveRescueModes = GetSummaryField(RunSummary,'MSLocalAdaptiveRescueModes');
    summary.MSLocalAdaptiveRescueCooldownModes = GetSummaryField(RunSummary,'MSLocalAdaptiveRescueCooldownModes');
    summary.MSLocalAdaptiveMeanRecentRate = GetSummaryField(RunSummary,'MSLocalAdaptiveMeanRecentRate');
    summary.MSLocalAdaptiveMeanStagnationFE = GetSummaryField(RunSummary,'MSLocalAdaptiveMeanStagnationFE');
    summary.ACOOriginalGenerated = GetSummaryField(RunSummary,'ACOOriginalGenerated');
    summary.ACOCategoryGenerated = GetSummaryField(RunSummary,'ACOCategoryGenerated');
    summary.ACOOriginalRetained = GetSummaryField(RunSummary,'ACOOriginalRetained');
    summary.ACOCategoryRetained = GetSummaryField(RunSummary,'ACOCategoryRetained');
    summary.ACODuplicateFiltered = GetSummaryField(RunSummary,'ACODuplicateFiltered');
    summary.ACOExistingFiltered = GetSummaryField(RunSummary,'ACOExistingFiltered');
    summary.ACORBFOriginalSelections = GetSummaryField(RunSummary,'ACORBFOriginalSelections');
    summary.ACORBFCategorySelections = GetSummaryField(RunSummary,'ACORBFCategorySelections');
    summary.ACORFOriginalSelections = GetSummaryField(RunSummary,'ACORFOriginalSelections');
    summary.ACORFCategorySelections = GetSummaryField(RunSummary,'ACORFCategorySelections');
    summary.ACORandomOriginalSelections = GetSummaryField(RunSummary,'ACORandomOriginalSelections');
    summary.ACORandomCategorySelections = GetSummaryField(RunSummary,'ACORandomCategorySelections');
    summary.ACORBFOriginalImprovements = GetSummaryField(RunSummary,'ACORBFOriginalImprovements');
    summary.ACORBFCategoryImprovements = GetSummaryField(RunSummary,'ACORBFCategoryImprovements');
    summary.ACORFOriginalImprovements = GetSummaryField(RunSummary,'ACORFOriginalImprovements');
    summary.ACORFCategoryImprovements = GetSummaryField(RunSummary,'ACORFCategoryImprovements');
    summary.ACORandomOriginalImprovements = GetSummaryField(RunSummary,'ACORandomOriginalImprovements');
    summary.ACORandomCategoryImprovements = GetSummaryField(RunSummary,'ACORandomCategoryImprovements');
    summary.ACOCategorySelectionRate = GetSummaryField(RunSummary,'ACOCategorySelectionRate');
    summary.MixLSAttempts = GetSummaryField(RunSummary,'MixLSAttempts');
    summary.MixLSBaselineCandidates = GetSummaryField(RunSummary,'MixLSBaselineCandidates');
    summary.MixLSDiscreteCandidates = GetSummaryField(RunSummary,'MixLSDiscreteCandidates');
    summary.MixLSTotalCandidates = GetSummaryField(RunSummary,'MixLSTotalCandidates');
    summary.MixLSSelectedBaseline = GetSummaryField(RunSummary,'MixLSSelectedBaseline');
    summary.MixLSSelectedMixed = GetSummaryField(RunSummary,'MixLSSelectedMixed');
    summary.MixLSSkipped = GetSummaryField(RunSummary,'MixLSSkipped');
    summary.MixLSMeanSelectedPred = GetSummaryField(RunSummary,'MixLSMeanSelectedPred');
    summary.MixLSMeanSelectedNovelty = GetSummaryField(RunSummary,'MixLSMeanSelectedNovelty');
    summary.MixLSMixedSelectionRate = GetSummaryField(RunSummary,'MixLSMixedSelectionRate');

    clear cleanup;
end


function summary = WriteInstanceStats(instanceDir, instanceName, seedSummaries)
    seedTable = struct2table(seedSummaries);
    writetable(seedTable, fullfile(instanceDir,sprintf('stats.%s.csv',instanceName)));
    save(fullfile(instanceDir,sprintf('stats.%s.mat',instanceName)), ...
        'seedSummaries','seedTable');

    summary = struct();
    summary.Instance = instanceName;
    summary.CompletedSeeds = height(seedTable);
    summary.MinBestObjective = min(seedTable.BestObjective);
    summary.MeanBestObjective = mean(seedTable.BestObjective);
    summary.StdBestObjective = std(seedTable.BestObjective);
    summary.MaxBestObjective = max(seedTable.BestObjective);
    summary.MeanRuntime = mean(seedTable.Runtime);
    summary.MeanTotalFE = mean(seedTable.TotalFE);

    filePath = fullfile(instanceDir,sprintf('stats.%s.txt',instanceName));
    fileID = fopen(filePath,'w');
    if fileID == -1
        error('RunEOPCCVExperiment:StatsFileOpenError', ...
            'Could not open %s for writing.', filePath);
    end
    cleaner = onCleanup(@() fclose(fileID));
    fprintf(fileID,'Instance %s\n',instanceName);
    fprintf(fileID,'Completed seeds %d\n',summary.CompletedSeeds);
    fprintf(fileID,'Min %.12g\n',summary.MinBestObjective);
    fprintf(fileID,'Mean %.12g\n',summary.MeanBestObjective);
    fprintf(fileID,'Std %.12g\n',summary.StdBestObjective);
    fprintf(fileID,'Max %.12g\n',summary.MaxBestObjective);
    fprintf(fileID,'MeanRuntime %.6f\n',summary.MeanRuntime);
    clear cleaner;
end


function value = ParseProblemId(problemId)
    if isnumeric(problemId)
        value = round(problemId);
    elseif isstring(problemId) || ischar(problemId)
        token = regexp(char(problemId),'^F?(\d+)$','tokens','once');
        if isempty(token)
            error('RunEOPCCVExperiment:InvalidProblemId', ...
                'Problem id must be a number or text like F1.');
        end
        value = str2double(token{1});
    else
        error('RunEOPCCVExperiment:InvalidProblemId', ...
            'Problem id must be numeric or text.');
    end

    if isnan(value) || value < 1 || value > 30
        error('RunEOPCCVExperiment:InvalidProblemId', ...
            'Problem id must be between 1 and 30.');
    end
end


function summary = EmptySeedSummary()
    summary = struct( ...
        'Seed', NaN, ...
        'BestObjective', NaN, ...
        'BestFE', NaN, ...
        'Runtime', NaN, ...
        'TotalFE', NaN, ...
        'RBFEvaluations', NaN, ...
        'RFEvaluations', NaN, ...
        'RandomEvaluations', NaN, ...
        'BasinHopEvaluations', NaN, ...
        'BasinHopExploreEvaluations', NaN, ...
        'LocalSearchEvaluations', NaN, ...
        'LocalSearchBestOnlyEvaluations', NaN, ...
        'LocalSearchMultiStartEvaluations', NaN, ...
        'LocalSearchCategoryEvaluations', NaN, ...
        'RBFImprovements', NaN, ...
        'RFImprovements', NaN, ...
        'RandomImprovements', NaN, ...
        'BasinHopImprovements', NaN, ...
        'BasinHopExploreImprovements', NaN, ...
        'LocalSearchImprovements', NaN, ...
        'LocalSearchBestOnlyImprovements', NaN, ...
        'LocalSearchMultiStartImprovements', NaN, ...
        'LocalSearchCategoryImprovements', NaN, ...
        'RBF_MAE', NaN, ...
        'RBF_RMSE', NaN, ...
        'RF_MAE', NaN, ...
        'RF_RMSE', NaN, ...
        'BasinHop_MAE', NaN, ...
        'BasinHop_RMSE', NaN, ...
        'BasinHopExplore_MAE', NaN, ...
        'BasinHopExplore_RMSE', NaN, ...
        'BasinTriggerAttempts', NaN, ...
        'BasinTriggers', NaN, ...
        'BasinVetoes', NaN, ...
        'BasinExploreTriggers', NaN, ...
        'BasinAlignmentSamples', NaN, ...
        'BasinAlignmentGuidedAttempts', NaN, ...
        'BasinCandidateVetoes', NaN, ...
        'BasinRandomFallbacks', NaN, ...
        'BasinIsRastrigin', NaN, ...
        'EnsembleEvaluations', NaN, ...
        'EnsembleImprovements', NaN, ...
        'ET_MAE', NaN, ...
        'ET_RMSE', NaN, ...
        'GBDT_MAE', NaN, ...
        'GBDT_RMSE', NaN, ...
        'Ensemble_MAE', NaN, ...
        'Ensemble_RMSE', NaN, ...
        'AlignmentSamples', NaN, ...
        'AlignmentGuidedEvaluations', NaN, ...
        'AlignmentFallbackEvaluations', NaN, ...
        'AlignmentError_MAE', NaN, ...
        'AlignmentError_RMSE', NaN, ...
        'MSLocalStartAttempts', NaN, ...
        'MSLocalSupportedCategories', NaN, ...
        'MSLocalCategoryStarts', NaN, ...
        'MSLocalDBStarts', NaN, ...
        'MSLocalACOStarts', NaN, ...
        'MSLocalSurrogateAttempts', NaN, ...
        'MSLocalCandidates', NaN, ...
        'MSLocalBestCandidates', NaN, ...
        'MSLocalMultiCandidates', NaN, ...
        'MSLocalCategoryCandidates', NaN, ...
        'MSLocalDBCandidates', NaN, ...
        'MSLocalACOCandidates', NaN, ...
        'MSLocalGatedCandidates', NaN, ...
        'MSLocalSuccesses', NaN, ...
        'MSLocalSwitches', NaN, ...
        'MSLocalFallbacks', NaN, ...
        'MSLocalSkippedByGate', NaN, ...
        'MSLocalMeanSelectedPred', NaN, ...
        'MSLocalMeanSelectedPredStart', NaN, ...
        'MSLocalMeanSelectedPredImprovement', NaN, ...
        'MSLocalMeanSelectedNovelty', NaN, ...
        'MSLocalMeanSelectedStartObjective', NaN, ...
        'MSLocalBestStartSelectionRate', NaN, ...
        'MSLocalCategorySelectionRate', NaN, ...
        'MSLocalMeanSelectedCategorySamples', NaN, ...
        'MSLocalACOStartSelectionRate', NaN, ...
        'MSLocalAdaptiveBestOnlyModes', NaN, ...
        'MSLocalAdaptiveMultiModes', NaN, ...
        'MSLocalAdaptiveStagnationTriggers', NaN, ...
        'MSLocalAdaptiveGoodRateTriggers', NaN, ...
        'MSLocalAdaptiveLowRateBlocks', NaN, ...
        'MSLocalAdaptiveRescueModes', NaN, ...
        'MSLocalAdaptiveRescueCooldownModes', NaN, ...
        'MSLocalAdaptiveMeanRecentRate', NaN, ...
        'MSLocalAdaptiveMeanStagnationFE', NaN, ...
        'ACOOriginalGenerated', NaN, ...
        'ACOCategoryGenerated', NaN, ...
        'ACOOriginalRetained', NaN, ...
        'ACOCategoryRetained', NaN, ...
        'ACODuplicateFiltered', NaN, ...
        'ACOExistingFiltered', NaN, ...
        'ACORBFOriginalSelections', NaN, ...
        'ACORBFCategorySelections', NaN, ...
        'ACORFOriginalSelections', NaN, ...
        'ACORFCategorySelections', NaN, ...
        'ACORandomOriginalSelections', NaN, ...
        'ACORandomCategorySelections', NaN, ...
        'ACORBFOriginalImprovements', NaN, ...
        'ACORBFCategoryImprovements', NaN, ...
        'ACORFOriginalImprovements', NaN, ...
        'ACORFCategoryImprovements', NaN, ...
        'ACORandomOriginalImprovements', NaN, ...
        'ACORandomCategoryImprovements', NaN, ...
        'ACOCategorySelectionRate', NaN, ...
        'MixLSAttempts', NaN, ...
        'MixLSBaselineCandidates', NaN, ...
        'MixLSDiscreteCandidates', NaN, ...
        'MixLSTotalCandidates', NaN, ...
        'MixLSSelectedBaseline', NaN, ...
        'MixLSSelectedMixed', NaN, ...
        'MixLSSkipped', NaN, ...
        'MixLSMeanSelectedPred', NaN, ...
        'MixLSMeanSelectedNovelty', NaN, ...
        'MixLSMixedSelectionRate', NaN);
end


function EnsureFolder(folderPath)
    if ~isfolder(folderPath)
        mkdir(folderPath);
    end
end


function value = ReadIntegerEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
        return;
    end

    value = str2double(rawValue);
    if isnan(value) || ~isfinite(value)
        error('RunEOPCCVExperiment:InvalidIntegerEnv', ...
            'Environment variable %s must be numeric.', name);
    end

    value = round(value);
end


function value = ReadBooleanEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
        return;
    end

    rawValue = lower(strtrim(rawValue));
    value = any(strcmp(rawValue,{'1','true','yes','on'}));
end


function value = ReadStringEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
    else
        value = rawValue;
    end
end


function value = GetSummaryField(RunSummary,fieldName)
    if isfield(RunSummary,fieldName)
        value = RunSummary.(fieldName);
    else
        value = nan;
    end
end


function tf = HasParallelSupport()
    tf = exist('gcp','file') == 2 && ...
        exist('parpool','file') == 2 && ...
        license('test','Distrib_Computing_Toolbox');
end


function pool = EnsureLocalPool(numWorkers)
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local',numWorkers);
        return;
    end

    if pool.NumWorkers ~= numWorkers
        delete(pool);
        pool = parpool('local',numWorkers);
    end
end


function WriteHistoryCsv(filePath, History)
    T = table(History.FE(:),History.Best(:), ...
        'VariableNames',{'FE','Best'});
    writetable(T,filePath);
end


function WriteEvalLogCsv(filePath, EvalLog)
    T = table(EvalLog.FE(:),EvalLog.Source(:),EvalLog.Obj(:), ...
        EvalLog.Best(:),logical(EvalLog.Improved(:)), ...
        'VariableNames',{'FE','Source','Obj','Best','Improved'});
    writetable(T,filePath);
end


function WriteSurLogCsv(filePath, SurLog)
    if isempty(SurLog.FE)
        T = table(zeros(0,1),cell(0,1),zeros(0,1),zeros(0,1),zeros(0,1), ...
            'VariableNames',{'FE','Source','Pred','Real','AbsError'});
    else
        absError = abs(SurLog.Pred(:) - SurLog.Real(:));
        T = table(SurLog.FE(:),SurLog.Source(:),SurLog.Pred(:), ...
            SurLog.Real(:),absError, ...
            'VariableNames',{'FE','Source','Pred','Real','AbsError'});
    end
    writetable(T,filePath);
end


function WriteAlignLogCsv(filePath, AlignLog)
    if isempty(AlignLog.FE)
        T = table(zeros(0,1),cell(0,1),false(0,1),zeros(0,1),zeros(0,1), ...
            zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1), ...
            'VariableNames',{'FE','Source','UsedAlignment','PredObj','RealObj', ...
            'AbsError','PredAbsError','Score','Quality','Novelty'});
    else
        T = table(AlignLog.FE(:),AlignLog.Source(:),logical(AlignLog.UsedAlignment(:)), ...
            AlignLog.PredObj(:),AlignLog.RealObj(:),AlignLog.AbsError(:), ...
            AlignLog.PredAbsError(:),AlignLog.Score(:),AlignLog.Quality(:), ...
            AlignLog.Novelty(:), ...
            'VariableNames',{'FE','Source','UsedAlignment','PredObj','RealObj', ...
            'AbsError','PredAbsError','Score','Quality','Novelty'});
    end
    writetable(T,filePath);
end


function WriteSolutionFile(filePath, bestObj, bestFE, bestX)
    fileID = fopen(filePath,'w');
    if fileID == -1
        error('RunEOPCCVExperiment:SolutionFileOpenError', ...
            'Could not open %s for writing.', filePath);
    end
    cleaner = onCleanup(@() fclose(fileID));
    fprintf(fileID,'BestObjective %.12g\n',bestObj);
    fprintf(fileID,'BestFE %d\n',bestFE);
    fprintf(fileID,'BestDecision');
    fprintf(fileID,' %.12g',bestX);
    fprintf(fileID,'\n');
    clear cleaner;
end


function WriteSeedSummaryFile(filePath, RunSummary)
    fileID = fopen(filePath,'w');
    if fileID == -1
        error('RunEOPCCVExperiment:SummaryFileOpenError', ...
            'Could not open %s for writing.', filePath);
    end
    cleaner = onCleanup(@() fclose(fileID));
    if isfield(RunSummary,'Algorithm')
        fprintf(fileID,'Algorithm %s\n',RunSummary.Algorithm);
    end
    if isfield(RunSummary,'BasinAblation')
        fprintf(fileID,'BasinAblation %s\n',RunSummary.BasinAblation);
        fprintf(fileID,'BasinUseArchive %d\n',RunSummary.BasinUseArchive);
        fprintf(fileID,'BasinAdaptiveOperators %d\n',RunSummary.BasinAdaptiveOperators);
        fprintf(fileID,'BasinRandomHopSelection %d\n',RunSummary.BasinRandomHopSelection);
    end
    if isfield(RunSummary,'BasinTriggers')
        WriteOptionalSummaryField(fileID,RunSummary,'BasinIsRastrigin','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinTriggerAttempts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinTriggers','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinVetoes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinExploreTriggers','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinAlignmentSamples','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinAlignmentGuidedAttempts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinCandidateVetoes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinRandomFallbacks','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinRastriginStagFE','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinOtherStagFE','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinPredImprovementTol','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinErrorMarginK','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinVetoQuantile','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinMinAlign','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinRBFAnomalyWeight','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinExploreAfterVetoes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinExploreCooldownFE','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'BasinExploreStagFE','%d');
    end
    if isfield(RunSummary,'MSLocalStartAttempts')
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalStartAttempts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalSupportedCategories','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalCategoryStarts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalDBStarts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalACOStarts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalSurrogateAttempts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalBestCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMultiCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalCategoryCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalDBCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalACOCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalGatedCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalSuccesses','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalSwitches','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalFallbacks','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalSkippedByGate','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMeanSelectedPred','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMeanSelectedPredStart','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMeanSelectedPredImprovement','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMeanSelectedNovelty','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMeanSelectedStartObjective','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalBestStartSelectionRate','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalCategorySelectionRate','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalMeanSelectedCategorySamples','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalACOStartSelectionRate','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveBestOnlyModes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveMultiModes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveStagnationTriggers','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveGoodRateTriggers','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveLowRateBlocks','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveRescueModes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveRescueCooldownModes','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveMeanRecentRate','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MSLocalAdaptiveMeanStagnationFE','%.12g');
    end
    if isfield(RunSummary,'ACOOriginalGenerated')
        WriteOptionalSummaryField(fileID,RunSummary,'ACOOriginalGenerated','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACOCategoryGenerated','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACOOriginalRetained','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACOCategoryRetained','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACODuplicateFiltered','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACOExistingFiltered','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORBFOriginalSelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORBFCategorySelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORFOriginalSelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORFCategorySelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORandomOriginalSelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORandomCategorySelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORBFOriginalImprovements','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORBFCategoryImprovements','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORFOriginalImprovements','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORFCategoryImprovements','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORandomOriginalImprovements','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACORandomCategoryImprovements','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'ACOCategorySelectionRate','%.12g');
    end
    if isfield(RunSummary,'MixLSAttempts')
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSAttempts','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSBaselineCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSDiscreteCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSTotalCandidates','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSSelectedBaseline','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSSelectedMixed','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSSkipped','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSMeanSelectedPred','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSMeanSelectedNovelty','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'MixLSMixedSelectionRate','%.12g');
    end
    if isfield(RunSummary,'SoftScoreRBFSelections')
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFSelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFSelections','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFFallbacks','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFFallbacks','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFChanged','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFChanged','%d');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFChangeRate','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFChangeRate','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFMeanPredRank','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFMeanPredRank','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFMeanEvidence','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFMeanEvidence','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFMeanNovelty','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFMeanNovelty','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFMeanUncertainty','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFMeanUncertainty','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRBFMeanSameCategoryCount','%.12g');
        WriteOptionalSummaryField(fileID,RunSummary,'SoftScoreRFMeanSameCategoryCount','%.12g');
    end
    fprintf(fileID,'Problem %s\n',RunSummary.Problem);
    fprintf(fileID,'Seed %d\n',RunSummary.Seed);
    fprintf(fileID,'BestObjective %.12g\n',RunSummary.BestObjective);
    fprintf(fileID,'BestFE %d\n',RunSummary.BestFE);
    fprintf(fileID,'Runtime %.6f\n',RunSummary.Runtime);
    fprintf(fileID,'TotalFE %d\n',RunSummary.TotalFE);
    fprintf(fileID,'RBFEvaluations %d\n',RunSummary.RBFEvaluations);
    fprintf(fileID,'RFEvaluations %d\n',RunSummary.RFEvaluations);
    fprintf(fileID,'RandomEvaluations %d\n',RunSummary.RandomEvaluations);
    if isfield(RunSummary,'BasinHopEvaluations')
        fprintf(fileID,'BasinHopEvaluations %d\n',RunSummary.BasinHopEvaluations);
    end
    if isfield(RunSummary,'BasinHopExploreEvaluations')
        fprintf(fileID,'BasinHopExploreEvaluations %d\n',RunSummary.BasinHopExploreEvaluations);
    end
    fprintf(fileID,'LocalSearchEvaluations %d\n',RunSummary.LocalSearchEvaluations);
    WriteOptionalSummaryField(fileID,RunSummary,'LocalSearchBestOnlyEvaluations','%d');
    WriteOptionalSummaryField(fileID,RunSummary,'LocalSearchMultiStartEvaluations','%d');
    WriteOptionalSummaryField(fileID,RunSummary,'LocalSearchCategoryEvaluations','%d');
    fprintf(fileID,'RBFImprovements %d\n',RunSummary.RBFImprovements);
    fprintf(fileID,'RFImprovements %d\n',RunSummary.RFImprovements);
    fprintf(fileID,'RandomImprovements %d\n',RunSummary.RandomImprovements);
    if isfield(RunSummary,'BasinHopImprovements')
        fprintf(fileID,'BasinHopImprovements %d\n',RunSummary.BasinHopImprovements);
    end
    if isfield(RunSummary,'BasinHopExploreImprovements')
        fprintf(fileID,'BasinHopExploreImprovements %d\n',RunSummary.BasinHopExploreImprovements);
    end
    fprintf(fileID,'LocalSearchImprovements %d\n',RunSummary.LocalSearchImprovements);
    WriteOptionalSummaryField(fileID,RunSummary,'LocalSearchBestOnlyImprovements','%d');
    WriteOptionalSummaryField(fileID,RunSummary,'LocalSearchMultiStartImprovements','%d');
    WriteOptionalSummaryField(fileID,RunSummary,'LocalSearchCategoryImprovements','%d');
    fprintf(fileID,'RBF_MAE %.12g\n',RunSummary.RBF_MAE);
    fprintf(fileID,'RBF_RMSE %.12g\n',RunSummary.RBF_RMSE);
    fprintf(fileID,'RF_MAE %.12g\n',RunSummary.RF_MAE);
    fprintf(fileID,'RF_RMSE %.12g\n',RunSummary.RF_RMSE);
    if isfield(RunSummary,'BasinHop_MAE')
        fprintf(fileID,'BasinHop_MAE %.12g\n',RunSummary.BasinHop_MAE);
        fprintf(fileID,'BasinHop_RMSE %.12g\n',RunSummary.BasinHop_RMSE);
    end
    if isfield(RunSummary,'BasinHopExplore_MAE')
        fprintf(fileID,'BasinHopExplore_MAE %.12g\n',RunSummary.BasinHopExplore_MAE);
        fprintf(fileID,'BasinHopExplore_RMSE %.12g\n',RunSummary.BasinHopExplore_RMSE);
    end
    if isfield(RunSummary,'EnsembleEvaluations')
        fprintf(fileID,'EnsembleEvaluations %d\n',RunSummary.EnsembleEvaluations);
        fprintf(fileID,'EnsembleImprovements %d\n',RunSummary.EnsembleImprovements);
    end
    if isfield(RunSummary,'ET_MAE')
        fprintf(fileID,'ET_MAE %.12g\n',RunSummary.ET_MAE);
        fprintf(fileID,'ET_RMSE %.12g\n',RunSummary.ET_RMSE);
    end
    if isfield(RunSummary,'GBDT_MAE')
        fprintf(fileID,'GBDT_MAE %.12g\n',RunSummary.GBDT_MAE);
        fprintf(fileID,'GBDT_RMSE %.12g\n',RunSummary.GBDT_RMSE);
    end
    if isfield(RunSummary,'Ensemble_MAE')
        fprintf(fileID,'Ensemble_MAE %.12g\n',RunSummary.Ensemble_MAE);
        fprintf(fileID,'Ensemble_RMSE %.12g\n',RunSummary.Ensemble_RMSE);
    end
    if isfield(RunSummary,'AlignmentSamples')
        fprintf(fileID,'AlignmentSamples %d\n',RunSummary.AlignmentSamples);
        fprintf(fileID,'AlignmentGuidedEvaluations %d\n',RunSummary.AlignmentGuidedEvaluations);
        fprintf(fileID,'AlignmentFallbackEvaluations %d\n',RunSummary.AlignmentFallbackEvaluations);
        fprintf(fileID,'AlignmentError_MAE %.12g\n',RunSummary.AlignmentError_MAE);
        fprintf(fileID,'AlignmentError_RMSE %.12g\n',RunSummary.AlignmentError_RMSE);
    end
    clear cleaner;
end


function WriteOptionalSummaryField(fileID, RunSummary, fieldName, valueFormat)
    if isfield(RunSummary,fieldName)
        fprintf(fileID,'%s ',fieldName);
        fprintf(fileID,valueFormat,RunSummary.(fieldName));
        fprintf(fileID,'\n');
    end
end
