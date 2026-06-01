% Run AGESV3 on an EOPCCV benchmark with PlatEMO base classes.
%
% Usage:
%   Open MATLAB, cd to this folder, then run:
%       run_agesv3_eopccv

clear; clc;

thisDir    = fileparts(mfilename('fullpath'));
platemoDir = fullfile(fileparts(thisDir), 'PlatEMO', 'PlatEMO');

addpath(genpath(platemoDir));
addpath(genpath(thisDir));

problem    = @EOPCCV_F12;
parentN    = 60;
offspringN = 100;
maxFE      = 600;

fprintf('Using AGESV3: %s\n',which('AGESV3'));
fprintf('Using problem: %s\n',which(func2str(problem)));

Problem   = problem('maxFE',maxFE);
Algorithm = AGESV3('parameter',{parentN,offspringN},'save',0);
Algorithm.Solve(Problem);

Population = Algorithm.result{end,2};
decs = Population.decs;
objs = Population.objs;

[bestObj,bestId] = min(objs);
bestX = decs(bestId,:);

fprintf('\nBest objective: %.12g\n',bestObj);
fprintf('Best decision vector:\n');
disp(bestX);
