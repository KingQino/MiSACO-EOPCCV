function instanceSummary = RunEOPCCVExperiment(problemId, seeds, maxFE, parentN, offspringN, statsRoot)
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

    fprintf('Running %s for seeds %s\n', problemClass, mat2str(seeds));
    fprintf('Detailed CSV/result logs: %d\n', saveDetailedLogs);
    seedSummaries = repmat(EmptySeedSummary(), numel(seeds), 1);

    for k = 1:numel(seeds)
        seed = seeds(k);
        seedDir = fullfile(instanceDir,num2str(seed));
        EnsureFolder(seedDir);
        seedSummaries(k) = RunSingleSeed(problemHandle, problemClass, ...
            instanceName, seed, maxFE, parentN, offspringN, seedDir, saveDetailedLogs);
    end

    instanceSummary = WriteInstanceStats(instanceDir, instanceName, seedSummaries);
    fprintf('Completed %s | best=%.12g mean=%.12g std=%.12g completed=%d\n', ...
        instanceName, instanceSummary.MinBestObjective, ...
        instanceSummary.MeanBestObjective, instanceSummary.StdBestObjective, ...
        instanceSummary.CompletedSeeds);
end


function summary = RunSingleSeed(problemHandle, problemClass, instanceName, seed, maxFE, parentN, offspringN, seedDir, saveDetailedLogs)
    historyFile = fullfile(seedDir, sprintf('history.%s.seed%d.mat', instanceName, seed));
    oldHistoryFile = getenv('MISACO_HISTORY_FILE');
    setenv('MISACO_HISTORY_FILE',historyFile);
    cleanup = onCleanup(@() setenv('MISACO_HISTORY_FILE',oldHistoryFile));

    rng(seed,'twister');
    fprintf('  Seed %d -> %s\n', seed, seedDir);

    Problem   = problemHandle('maxFE',maxFE);
    Algorithm = MiSACO('parameter',{parentN,offspringN},'save',0);
    Algorithm.Solve(Problem);

    Population = Algorithm.result{end,2};
    popDecs = Population.decs;
    popObjs = Population.objs;
    [bestObj,bestId] = min(popObjs);
    bestX = popDecs(bestId,:);

    if ~isfile(historyFile)
        error('RunEOPCCVExperiment:MissingHistoryFile', ...
            'MiSACO did not write the expected history file: %s', historyFile);
    end

    load(historyFile,'History','EvalLog','SurLog','RunSummary');
    RunSummary.Problem      = problemClass;
    RunSummary.Instance     = instanceName;
    RunSummary.Seed         = seed;
    RunSummary.MaxFE        = maxFE;
    RunSummary.ParentN      = parentN;
    RunSummary.OffspringN   = offspringN;
    RunSummary.BestDecision = bestX;
    RunSummary.SaveDetailedLogs = saveDetailedLogs;
    save(historyFile,'History','EvalLog','SurLog','RunSummary');

    WriteHistoryCsv(fullfile(seedDir,sprintf('evols.%s.csv',instanceName)),History);
    WriteSolutionFile(fullfile(seedDir,sprintf('solution.%s.txt',instanceName)), ...
        bestObj, RunSummary.BestFE, bestX);
    WriteSeedSummaryFile(fullfile(seedDir,sprintf('summary.%s.seed%d.txt',instanceName,seed)),RunSummary);

    if saveDetailedLogs
        WriteEvalLogCsv(fullfile(seedDir,sprintf('eval_log.%s.csv',instanceName)),EvalLog);
        WriteSurLogCsv(fullfile(seedDir,sprintf('surrogate_log.%s.csv',instanceName)),SurLog);

        result = struct();
        result.RunSummary = RunSummary;
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
    summary.LocalSearchEvaluations = RunSummary.LocalSearchEvaluations;
    summary.RBFImprovements = RunSummary.RBFImprovements;
    summary.RFImprovements = RunSummary.RFImprovements;
    summary.RandomImprovements = RunSummary.RandomImprovements;
    summary.LocalSearchImprovements = RunSummary.LocalSearchImprovements;
    summary.RBF_MAE = RunSummary.RBF_MAE;
    summary.RBF_RMSE = RunSummary.RBF_RMSE;
    summary.RF_MAE = RunSummary.RF_MAE;
    summary.RF_RMSE = RunSummary.RF_RMSE;

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
        'LocalSearchEvaluations', NaN, ...
        'RBFImprovements', NaN, ...
        'RFImprovements', NaN, ...
        'RandomImprovements', NaN, ...
        'LocalSearchImprovements', NaN, ...
        'RBF_MAE', NaN, ...
        'RBF_RMSE', NaN, ...
        'RF_MAE', NaN, ...
        'RF_RMSE', NaN);
end


function EnsureFolder(folderPath)
    if ~isfolder(folderPath)
        mkdir(folderPath);
    end
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
    fprintf(fileID,'Problem %s\n',RunSummary.Problem);
    fprintf(fileID,'Seed %d\n',RunSummary.Seed);
    fprintf(fileID,'BestObjective %.12g\n',RunSummary.BestObjective);
    fprintf(fileID,'BestFE %d\n',RunSummary.BestFE);
    fprintf(fileID,'Runtime %.6f\n',RunSummary.Runtime);
    fprintf(fileID,'TotalFE %d\n',RunSummary.TotalFE);
    fprintf(fileID,'RBFEvaluations %d\n',RunSummary.RBFEvaluations);
    fprintf(fileID,'RFEvaluations %d\n',RunSummary.RFEvaluations);
    fprintf(fileID,'RandomEvaluations %d\n',RunSummary.RandomEvaluations);
    fprintf(fileID,'LocalSearchEvaluations %d\n',RunSummary.LocalSearchEvaluations);
    fprintf(fileID,'RBFImprovements %d\n',RunSummary.RBFImprovements);
    fprintf(fileID,'RFImprovements %d\n',RunSummary.RFImprovements);
    fprintf(fileID,'RandomImprovements %d\n',RunSummary.RandomImprovements);
    fprintf(fileID,'LocalSearchImprovements %d\n',RunSummary.LocalSearchImprovements);
    fprintf(fileID,'RBF_MAE %.12g\n',RunSummary.RBF_MAE);
    fprintf(fileID,'RBF_RMSE %.12g\n',RunSummary.RBF_RMSE);
    fprintf(fileID,'RF_MAE %.12g\n',RunSummary.RF_MAE);
    fprintf(fileID,'RF_RMSE %.12g\n',RunSummary.RF_RMSE);
    clear cleaner;
end
