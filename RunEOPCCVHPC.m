function RunEOPCCVHPC(varargin)
%RunEOPCCVHPC Run one EOPCCV problem selected by a Slurm array task.

    projectRoot = fileparts(mfilename('fullpath'));
    platemoDir  = ReadStringEnv('MISACO_PLATEMO_DIR', ...
        fullfile(fileparts(projectRoot),'PlatEMO','PlatEMO'));

    if ~isfolder(platemoDir)
        error('RunEOPCCVHPC:PlatEMONotFound', ...
            'PlatEMO directory does not exist: %s', platemoDir);
    end

    addpath(genpath(platemoDir));
    addpath(genpath(projectRoot));

    arrayTaskId = [];
    if nargin >= 1
        arrayTaskId = ParseOptionalInteger(varargin{1}, 'arrayTaskId');
    end
    if isempty(arrayTaskId)
        arrayTaskId = ReadIntegerEnv('SLURM_ARRAY_TASK_ID', []);
    end
    if isempty(arrayTaskId)
        error('RunEOPCCVHPC:MissingArrayTaskId', ...
            ['SLURM_ARRAY_TASK_ID is not set. Pass arrayTaskId explicitly, ' ...
            'for example RunEOPCCVHPC(0).']);
    end

    arrayBase   = ReadIntegerEnv('MISACO_ARRAY_BASE', 0);
    maxTrials  = ReadIntegerEnv('MISACO_MAX_TRIALS', 10);
    maxFE      = ReadIntegerEnv('MISACO_MAX_FE', 600);
    parentN    = ReadIntegerEnv('MISACO_PARENT_N', 60);
    offspringN = ReadIntegerEnv('MISACO_OFFSPRING_N', 100);
    algorithmName = ReadStringEnv('MISACO_ALGORITHM','MiSACO');
    statsRoot  = ReadStringEnv('MISACO_STATS_ROOT', ...
        fullfile(projectRoot,algorithmName,'stats'));
    arrayMode  = lower(ReadStringEnv('MISACO_ARRAY_MODE','seed'));
    problemIds = 1:30;

    switch arrayMode
        case 'seed'
            taskIndex = arrayTaskId - arrayBase;
            totalTasks = numel(problemIds) * maxTrials;
            if taskIndex < 0 || taskIndex >= totalTasks
                error('RunEOPCCVHPC:ArrayIndexOutOfRange', ...
                    'Array task id %d is out of range for %d seed tasks.', ...
                    arrayTaskId, totalTasks);
            end
            problemId = problemIds(floor(taskIndex/maxTrials) + 1);
            seed = mod(taskIndex,maxTrials) + 1;
            oldWriteStats = getenv('MISACO_WRITE_INSTANCE_STATS');
            setenv('MISACO_WRITE_INSTANCE_STATS','0');
            cleanup = onCleanup(@() setenv('MISACO_WRITE_INSTANCE_STATS',oldWriteStats));

            maxNumCompThreads(1);
            fprintf('Array task %d selected EOPCCV_F%d seed %d\n', ...
                arrayTaskId, problemId, seed);
            fprintf('Algorithm: %s\n', algorithmName);
            fprintf('Stats root: %s\n', statsRoot);
            RunEOPCCVExperiment(problemId, seed, maxFE, parentN, offspringN, statsRoot, 1, algorithmName);
            clear cleanup;

        case 'problem'
            problemIndex = arrayTaskId - arrayBase + 1;
            if problemIndex < 1 || problemIndex > numel(problemIds)
                error('RunEOPCCVHPC:ArrayIndexOutOfRange', ...
                    'Array task id %d is out of range for base %d and %d problems.', ...
                    arrayTaskId, arrayBase, numel(problemIds));
            end
            problemId = problemIds(problemIndex);
            numWorkers = ReadIntegerEnv('MISACO_NUM_WORKERS', ...
                ReadIntegerEnv('SLURM_CPUS_PER_TASK', 1));
            numWorkers = max(1,min(numWorkers,maxTrials));

            maxNumCompThreads(1);
            fprintf('Array task %d selected EOPCCV_F%d\n', arrayTaskId, problemId);
            fprintf('Seeds: 1-%d, workers: %d\n', maxTrials, numWorkers);
            fprintf('Algorithm: %s\n', algorithmName);
            fprintf('Stats root: %s\n', statsRoot);
            RunEOPCCVExperiment(problemId, 1:maxTrials, maxFE, parentN, offspringN, statsRoot, numWorkers, algorithmName);

        otherwise
            error('RunEOPCCVHPC:InvalidArrayMode', ...
                'MISACO_ARRAY_MODE must be seed or problem, got %s.', arrayMode);
    end
end


function value = ParseOptionalInteger(rawValue, name)
    if isempty(rawValue)
        value = [];
        return;
    end

    if isnumeric(rawValue)
        value = round(rawValue);
        return;
    end

    if isstring(rawValue) || ischar(rawValue)
        value = str2double(rawValue);
    else
        error('RunEOPCCVHPC:InvalidOptionalInput', ...
            '%s must be numeric or text containing a number.', name);
    end

    if isnan(value) || ~isfinite(value)
        error('RunEOPCCVHPC:InvalidOptionalInput', ...
            '%s must be numeric.', name);
    end

    value = round(value);
end


function value = ReadIntegerEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
        return;
    end

    value = str2double(rawValue);
    if isnan(value) || ~isfinite(value)
        error('RunEOPCCVHPC:InvalidIntegerEnv', ...
            'Environment variable %s must be numeric.', name);
    end

    value = round(value);
end


function value = ReadStringEnv(name, defaultValue)
    rawValue = getenv(name);
    if isempty(rawValue)
        value = defaultValue;
    else
        value = rawValue;
    end
end
