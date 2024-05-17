function lambda = solve_quadratic(proj_dl, d_lin, H, g, delta)
%SOLVE_QUADRATIC Calculates the scalar that realizes the optimal convex
%combination of the projected dogleg point and the linearzied solution as
%an inexact solution of the constrained simple trust region subproblem
%
% Inputs:
%   - proj_dl: Dogleg point projected onto the standard simplex (vector)
%   - d_lin:   Solution of the linearized subproblem (vector)
%   - H:       Current iterate's approximation of the Hessian (matrix)
%   - g:       Current iterate's gradient (vector)
%   - delta:   Current trust region radius (positive scalar)
%
% Outputs:
%   - lambda: Scalar that realizes the optimal convex combination (scalar between 0 and 1)

% If proj_dl and d_lin coincide, skip all of the following
if isequal(proj_dl, d_lin)
    lambda = 0;
else 
    %% Compute zeros of quadratic constraint |proj_dl + lambda*(d_lin - proj_dl)|^2 - delta^2
    alpha = norm(d_lin - proj_dl)^2; % Always strictly positive, since proj_dl ~= d_lin
    beta = 2 * proj_dl' * (d_lin - proj_dl);
    gamma = norm(proj_dl)^2 - delta^2; % Always non-positive, since |proj_dl| <= delta
    
    % Mitternachtsformel
    lambda1 = (-beta - sqrt(beta^2 - 4*alpha*gamma)) / (2*alpha);
    lambda2 = (-beta + sqrt(beta^2 - 4*alpha*gamma)) / (2*alpha); % It always holds that lambda1 <= lambda2

    % Collect bounds for the lambda
    lb = max(0, lambda1);
    ub = min(1, lambda2);
    
    %% Compute values of quadratic function q(proj_dl + lambda*(d_lin-proj_dl))
    a = (d_lin - proj_dl)' * H * (d_lin - proj_dl);
    b = (g + H * proj_dl)' * (d_lin - proj_dl);
    c = g' * proj_dl + proj_dl' * H * proj_dl;

    vert = -b/a;
    if a > 0 % Quadratic is a parabola opened upwards
        if vert > ub
            lambda = ub;
        elseif vert < lb
            lambda = lb;
        else
            lambda = vert;
        end
    elseif a < 0 % Quadratic is a parabola opened downwards (this case should not occur!)
        warning("Check definiteness of H!");
        if b <= -a/2
            lambda = ub;
        else
            lambda = lb;
        end
    elseif a == 0 % This can only happen, if H is not positive definite
        warning("Check definiteness of H!")
    end
end
