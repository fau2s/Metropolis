import numpy as np
from scipy.optimize import least_squares

def est_mu_linreg( rho ):
    if (np.isnan(np.sum(rho))):
        mu = np.NaN
        return mu

    x0 = np.ones(2)
    #coeff = robustfit(rho[2:rho.size], rho[1:rho.size-1])
    coeff = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(rho[1:rho.size-1], rho[0:rho.size-2]))
    mu = coeff.x[1] / (1-coeff.x[0])
    # else:
    #        mu = np.mean(rho)

    return mu

def fun(x, rho, y):
    return x[0] * rho + x[1] - y


# function [ mu ] = est_mu_linreg( rho )
#     if (size(rho, 1) == 1)
#         rho = rho';
#     end
#     if (isnan(sum(rho)))
#         mu = NaN;
#         return;
#     end
#
#     %[coeff stats] = robustfit(rho(2:numel(rho)), rho(1:numel(rho)-1));
#     %if (abs(1 - max(rho(2:numel(rho)) ./ rho(1:numel(rho)-1))) > 0.012)
#     %if (stats.s < 25)
#         coeff = robustfit(rho(2:numel(rho)), rho(1:numel(rho)-1));
#         mu = coeff(1) / (1-coeff(2));
#         % figure(1); hold on; plot(rho ./ mu);
#     %{
#     else
#         mu = mean(rho);
#         figure(2); hold on; plot(rho ./ mu);
#     end
#     %}
# end