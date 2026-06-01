function [y, sigma2] = RBFInterp_Hamming(x,len_c, up, dn, para)
[~,D] = size(x);
ax = para.nodes;
nx = size(x, 1);
np = size(ax, 1);    

ymin = para.ymin;
ymax = para.ymax;
yrange = ymax - ymin;
yrange(yrange < eps) = eps;

ax_nc = ax(:,1:D-len_c);
ax_c  = ax(:,D-len_c+1:D);

x_nc = x(:,1:D-len_c);
xrange = up - dn;
xrange(xrange < eps) = eps;
x_nc = (x_nc - dn)./xrange - 0.5; 
x_c  = x(:,D-len_c+1:D);

x = [x_nc,x_c];

dis_euc = sum(abs(x_nc - ax_nc),2)';
dis_gower = sum( ax_c~= x_c ,2)';

r = dis_gower + dis_euc;
switch para.kernel
    case 'gaussian'
        Phi = radbas(sqrt(-log(.5))*r);
    case 'cubic'
        Phi = r.^3;
end

y = Phi * para.alpha;
%renormalization
y = repmat(yrange, nx, 1)./2 .* (y + 1) + repmat(ymin, nx, 1);

if nargout > 1
    switch para.kernel
        case 'gaussian'
            sigma2 = 1 / sqrt(2 * pi) - Phi*(pinv(para.Phi)*(Phi'));
        case 'cubic'
            sigma2 =  - Phi*(pinv(para.Phi)*(Phi'));
    end
    sigma2 = abs(diag(sigma2));
end

end
