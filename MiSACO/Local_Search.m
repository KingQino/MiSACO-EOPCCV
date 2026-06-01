function [out_best, success] = Local_Search(best_Arc_r,best_Arc_f,local_up,local_dn,best_r)
success  = false;
out_best = best_r;

local_up = local_up(:)';
local_dn = local_dn(:)';
if isempty(best_r) || any(~isfinite(best_r))
    best_r = local_dn + (local_up - local_dn).*rand(1,length(local_up));
end
best_r = min(max(best_r(:)',local_dn),local_up);
out_best = best_r;
freeDim = (local_up - local_dn) > 1e-6;
if ~any(freeDim)
    return;
end

if isempty(best_Arc_r) || isempty(best_Arc_f)
    return;
end

if size(best_Arc_f,1) == 1 && numel(best_Arc_f) == size(best_Arc_r,1)
    best_Arc_f = best_Arc_f(:);
else
    best_Arc_f = best_Arc_f(:,1);
end

valid = all(isfinite(best_Arc_r),2) & isfinite(best_Arc_f);
best_Arc_r = best_Arc_r(valid,:);
best_Arc_f = best_Arc_f(valid,:);

[~,distinct] = unique(round(best_Arc_r*1e10)/1e10,'rows');
best_Arc_r = best_Arc_r(distinct,:);
best_Arc_f = best_Arc_f(distinct,:);

if size(best_Arc_r,1) < size(best_Arc_r,2) + 2
    return;
end

try
    rbfnet_best = RBFCreate(best_Arc_r,best_Arc_f,'cubic');
catch err
    warning('Local_Search:RBFCreateFailed','Local search skipped because RBFCreate failed: %s',err.message);
    return;
end

[~,validStart] = surrogateObj(best_r(freeDim)',false);
if ~validStart
    return;
end

options = optimset('Algorithm','interior-point','MaxFunEvals',300,'Display','off','FinDiffRelStep',1e-8);
try
    candidateFree = fmincon(@safeObj,best_r(freeDim)',[],[],[],[],local_dn(freeDim)',local_up(freeDim)',[],options);
catch err
    warning('Local_Search:fminconFailed','Local search skipped because fmincon failed: %s',err.message);
    return;
end

candidate = best_r;
candidate(freeDim) = candidateFree(:)';
candidate = min(max(candidate,local_dn),local_up);
[~,validCandidate] = surrogateObj(candidate(freeDim)',false);
if all(isfinite(candidate)) && validCandidate
    out_best = candidate;
    success  = true;
end

    function y = safeObj(x)
        [y,validValue] = surrogateObj(x,true);
        if ~validValue
            y = 1e100;
        end
    end

    function [y,validValue] = surrogateObj(x,usePenalty)
        try
            fullX = best_r;
            fullX(freeDim) = x(:)';
            y = RBFInterp(fullX,rbfnet_best);
            y = y(1);
        catch
            y = nan;
        end
        validValue = isfinite(y);
        if ~validValue && usePenalty
            y = 1e100;
        end
    end
end
