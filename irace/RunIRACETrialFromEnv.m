function cost = RunIRACETrialFromEnv()
%RunIRACETrialFromEnv Evaluate one IRACE configuration on one fixed F:seed instance.

    projectRoot = ReadStringEnv('MISACO_PROJECT_ROOT', ...
        fileparts(fileparts(mfilename('fullpath'))));
    platemoDir = ReadStringEnv('MISACO_PLATEMO_DIR', ...
        fullfile(fileparts(projectRoot),'PlatEMO','PlatEMO'));

    addpath(genpath(projectRoot));
    if isfolder(platemoDir)
        addpath(genpath(platemoDir));
    else
        error('RunIRACETrialFromEnv:PlatEMONotFound', ...
            'PlatEMO directory does not exist: %s', platemoDir);
    end

    costFile = ReadStringEnv('MISACO_IRACE_COST_FILE','');
    try
        instanceSpec = ReadStringEnv('MISACO_IRACE_INSTANCE','');
        [problemId, seed] = ParseIRACEInstance(instanceSpec);

        params = ParseNumericVector(ReadStringEnv('MISACO_IRACE_PARAMS',''));
        if isempty(params)
            params = DefaultMiSACOMSLS5Params();
        end
        if numel(params) ~= 23
            error('RunIRACETrialFromEnv:InvalidParameterCount', ...
                'Expected 23 MiSACOMSLS5 parameters, got %d.', numel(params));
        end

        maxFE = ReadIntegerEnv('MISACO_IRACE_MAX_FE', ...
            ReadIntegerEnv('MISACO_MAX_FE',600));
        algorithmName = ReadStringEnv('MISACO_IRACE_ALGORITHM', ...
            ReadStringEnv('MISACO_ALGORITHM','MiSACOMSLS5'));
        statsRoot = ReadStringEnv('MISACO_IRACE_STATS_ROOT', ...
            fullfile(projectRoot,'irace','runs'));
        baselineRoot = ReadStringEnv('MISACO_IRACE_BASELINE_ROOT', ...
            fullfile(projectRoot,'MiSACOMSLS5','stats_01'));
        tau = ReadDoubleEnv('MISACO_IRACE_TAU',1e-6);

        configId = ReadStringEnv('MISACO_IRACE_CONFIG_ID','unknown_config');
        instanceId = ReadStringEnv('MISACO_IRACE_INSTANCE_ID','unknown_instance');
        iraceSeed = ReadStringEnv('MISACO_IRACE_SEED','unknown_seed');
        runRoot = fullfile(statsRoot,SanitizePathPart(configId), ...
            SanitizePathPart(instanceId),SanitizePathPart(iraceSeed));

        oldDetailed = getenv('MISACO_SAVE_DETAILED_LOGS');
        oldWriteStats = getenv('MISACO_WRITE_INSTANCE_STATS');
        oldParams = getenv('MISACO_ALGORITHM_PARAMS');
        setenv('MISACO_SAVE_DETAILED_LOGS','0');
        setenv('MISACO_WRITE_INSTANCE_STATS','0');
        setenv('MISACO_ALGORITHM_PARAMS',sprintf('%.17g ',params));
        cleanup = onCleanup(@() RestoreEnv(oldDetailed,oldWriteStats,oldParams));

        RunEOPCCVExperiment(problemId,seed,maxFE,params(1),params(2), ...
            runRoot,1,algorithmName,params);
        candidateObj = ReadCandidateObjective(runRoot,problemId,seed);
        baselineObj = ReadBaselineObjective(baselineRoot,problemId,seed);

        cost = log((candidateObj + tau)./(baselineObj + tau));
        if ~isfinite(cost)
            cost = 1e6;
        end

        fprintf('IRACE_RESULT instance=F%d:%d candidate=%.16g baseline=%.16g tau=%.16g cost=%.16g\n', ...
            problemId,seed,candidateObj,baselineObj,tau,cost);
        WriteCostFile(costFile,cost);
        clear cleanup;
    catch err
        cost = 1e6;
        fprintf(2,'IRACE_TRIAL_ERROR: %s\n',err.message);
        for k = 1:numel(err.stack)
            fprintf(2,'  at %s:%d\n',err.stack(k).file,err.stack(k).line);
        end
        WriteCostFile(costFile,cost);
    end
end


function value = ReadCandidateObjective(runRoot,problemId,seed)
    instanceName = sprintf('F%d',problemId);
    historyFile = fullfile(runRoot,instanceName,num2str(seed), ...
        sprintf('history.%s.seed%d.mat',instanceName,seed));
    if ~isfile(historyFile)
        error('RunIRACETrialFromEnv:MissingCandidateHistory', ...
            'Missing candidate history file: %s', historyFile);
    end

    data = load(historyFile);
    if isfield(data,'RunSummary') && isfield(data.RunSummary,'BestObjective')
        value = data.RunSummary.BestObjective;
    elseif isfield(data,'EvalLog') && isfield(data.EvalLog,'Obj') && ...
            ~isempty(data.EvalLog.Obj)
        value = min(data.EvalLog.Obj(:));
    else
        error('RunIRACETrialFromEnv:InvalidCandidateHistory', ...
            'Candidate history file does not contain RunSummary.BestObjective or EvalLog.Obj: %s', ...
            historyFile);
    end

    if ~isfinite(value)
        error('RunIRACETrialFromEnv:InvalidCandidateObjective', ...
            'Candidate objective is not finite for F%d seed %d.', problemId, seed);
    end
end


function params = DefaultMiSACOMSLS5Params()
    params = [60,100,6,30,5,30,0.01,0.005,0.08,0.20,12, ...
        0.45,0.15,30,60,0.70,60,0.10,0.35,0.05,0.02,3,45];
end


function [problemId, seed] = ParseIRACEInstance(instanceSpec)
    instanceSpec = strtrim(char(instanceSpec));
    token = regexp(instanceSpec,'^F?(\d+)\s*[:_,/-]\s*(?:seed)?\s*(\d+)$', ...
        'tokens','once');
    if isempty(token)
        error('RunIRACETrialFromEnv:InvalidInstance', ...
            'IRACE instance must look like F12:3 or 12:3, got "%s".', instanceSpec);
    end

    problemId = str2double(token{1});
    seed = str2double(token{2});
    if ~isfinite(problemId) || problemId < 1 || problemId > 30 || ...
            ~isfinite(seed) || seed < 1
        error('RunIRACETrialFromEnv:InvalidInstance', ...
            'Invalid problem/seed in IRACE instance "%s".', instanceSpec);
    end
    problemId = round(problemId);
    seed = round(seed);
end


function value = ReadBaselineObjective(baselineRoot,problemId,seed)
    instanceName = sprintf('F%d',problemId);
    csvFile = fullfile(baselineRoot,instanceName,sprintf('stats.%s.csv',instanceName));
    if ~isfile(csvFile)
        error('RunIRACETrialFromEnv:MissingBaseline', ...
            'Missing baseline CSV: %s', csvFile);
    end

    T = readtable(csvFile);
    idx = find(T.Seed == seed,1,'first');
    if isempty(idx)
        error('RunIRACETrialFromEnv:MissingBaselineSeed', ...
            'Baseline CSV %s does not contain seed %d.', csvFile, seed);
    end

    value = T.BestObjective(idx);
    if ~isfinite(value)
        error('RunIRACETrialFromEnv:InvalidBaseline', ...
            'Baseline objective is not finite for F%d seed %d.', problemId, seed);
    end
end


function value = ParseNumericVector(rawValue)
    rawValue = strtrim(char(rawValue));
    if isempty(rawValue)
        value = [];
        return;
    end

    rawValue = strrep(rawValue,',',' ');
    value = sscanf(rawValue,'%f').';
    if isempty(value) || any(~isfinite(value))
        error('RunIRACETrialFromEnv:InvalidNumericVector', ...
            'Could not parse numeric parameter vector "%s".', rawValue);
    end
end


function value = ReadIntegerEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
        return;
    end
    value = str2double(rawValue);
    if ~isfinite(value)
        error('RunIRACETrialFromEnv:InvalidIntegerEnv', ...
            '%s must be numeric.', name);
    end
    value = round(value);
end


function value = ReadDoubleEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
        return;
    end
    value = str2double(rawValue);
    if ~isfinite(value)
        error('RunIRACETrialFromEnv:InvalidDoubleEnv', ...
            '%s must be numeric.', name);
    end
end


function value = ReadStringEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
    else
        value = rawValue;
    end
end


function value = SanitizePathPart(value)
    value = char(value);
    value = regexprep(value,'[^A-Za-z0-9_.-]','_');
    if isempty(value)
        value = 'unknown';
    end
end


function WriteCostFile(costFile,cost)
    if isempty(costFile)
        return;
    end
    folderPath = fileparts(costFile);
    if ~isempty(folderPath) && ~isfolder(folderPath)
        mkdir(folderPath);
    end
    fileID = fopen(costFile,'w');
    if fileID == -1
        error('RunIRACETrialFromEnv:CostFileOpenError', ...
            'Could not open cost file: %s', costFile);
    end
    cleaner = onCleanup(@() fclose(fileID));
    fprintf(fileID,'%.16g\n',cost);
    clear cleaner;
end


function RestoreEnv(oldDetailed,oldWriteStats,oldParams)
    setenv('MISACO_SAVE_DETAILED_LOGS',oldDetailed);
    setenv('MISACO_WRITE_INSTANCE_STATS',oldWriteStats);
    setenv('MISACO_ALGORITHM_PARAMS',oldParams);
end
