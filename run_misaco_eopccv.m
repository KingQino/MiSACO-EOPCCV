% Run the local MiSACO on an EOPCCV benchmark with PlatEMO base classes.
%
% Usage:
%   Open MATLAB, cd to this folder, then run:
%       run_misaco_eopccv

clear; clc;

thisDir    = fileparts(mfilename('fullpath'));
platemoDir = fullfile(fileparts(thisDir), 'PlatEMO', 'PlatEMO');

addpath(genpath(platemoDir));
addpath(genpath(thisDir));

problem   = @EOPCCV_F1;
parentN   = 60;
offspringN = 100;

% Use parentN for a smoke test, or a larger value for a real run.
maxFE = 600;

fprintf('Using MiSACO: %s\n',which('MiSACO'));
fprintf('Using problem: %s\n',which(func2str(problem)));

Problem   = problem('maxFE',maxFE);
Algorithm = MiSACO('parameter',{parentN,offspringN},'save',0);
Algorithm.Solve(Problem);

Population = Algorithm.result{end,2};
decs = Population.decs;
objs = Population.objs;
cons = Population.cons;

[bestObj, bestId] = min(objs);
bestX = decs(bestId, :);

fprintf('\nBest objective: %.12g\n', bestObj);
fprintf('Best decision vector:\n');
disp(bestX);
