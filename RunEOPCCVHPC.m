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
    problemIds  = 1:30;
    problemIndex = arrayTaskId - arrayBase + 1;
    if problemIndex < 1 || problemIndex > numel(problemIds)
        error('RunEOPCCVHPC:ArrayIndexOutOfRange', ...
            'Array task id %d is out of range for base %d and %d problems.', ...
            arrayTaskId, arrayBase, numel(problemIds));
    end

    problemId  = problemIds(problemIndex);
    maxTrials  = ReadIntegerEnv('MISACO_MAX_TRIALS', 10);
    maxFE      = ReadIntegerEnv('MISACO_MAX_FE', 600);
    parentN    = ReadIntegerEnv('MISACO_PARENT_N', 60);
    offspringN = ReadIntegerEnv('MISACO_OFFSPRING_N', 100);
    statsRoot  = ReadStringEnv('MISACO_STATS_ROOT', ...
        fullfile(projectRoot,'MiSACO','stats'));

    maxNumCompThreads(1);
    fprintf('Array task %d selected EOPCCV_F%d\n', arrayTaskId, problemId);
    fprintf('Stats root: %s\n', statsRoot);

    RunEOPCCVExperiment(problemId, 1:maxTrials, maxFE, parentN, offspringN, statsRoot);
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
