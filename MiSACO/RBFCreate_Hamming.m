function para = RBFCreate_Hamming(ax, ay, len_c, up, dn, kernel)
[N, D] = size(ax);

ax_nc = ax(:,1:D-len_c);
xrange = up - dn;
xrange(xrange < eps) = eps;
ax_nc = (ax_nc - dn)./xrange - 0.5; 
ax_c  = ax(:,D-len_c+1:D);

ymin = min(ay, [], 1);
ymax = max(ay, [], 1);
yrange = ymax - ymin;
yrange(yrange < eps) = eps;
ay = 2./repmat(yrange, N, 1) .* (ay - repmat(ymin, N, 1)) - 1;

ax = [ax_nc,ax_c];

for i = 1:N 
    dis_gower(i,:) = sum( ax_c~= ax_c(i,:) ,2)';
    dis_euc(i,:) = sum( abs(ax_nc - ax_nc(i,:)),2)';
end

r = dis_gower + dis_euc;
switch kernel
    case 'gaussian'
        Phi = radbas(sqrt(-log(.5))*r);
    case 'cubic'
        Phi = r.^3;
end
[Phi,theta] = SolveRBFSystem(Phi,ay);
para.alpha = theta(1 : N, :);
para.ymin = ymin;
para.ymax = ymax;
para.nodes = ax;
para.kernel = kernel;
para.Phi = Phi;
end

function [Phi,theta] = SolveRBFSystem(Phi,ay)
N = size(Phi,1);
lambda = 1e-10*max(1,mean(abs(diag(Phi))));
PhiReg = Phi + lambda*eye(N);
conditionEstimate = rcond(PhiReg);

while conditionEstimate <= 1e-12 && lambda < 1e-3
    lambda = lambda*10;
    PhiReg = Phi + lambda*eye(N);
    conditionEstimate = rcond(PhiReg);
end

if conditionEstimate <= 1e-12
    theta = pinv(PhiReg)*ay;
else
    theta = PhiReg\ay;
end
Phi = PhiReg;
end
