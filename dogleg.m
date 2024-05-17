function [s, predicted_reduction] = dogleg(H, H_posdef, grad, delta)
%DOGLEG Calculates an inexact solution of the simple (non-constrained)
%trust region subproblem given some approximation of the Hessian and some
%gradient
%
% Inputs:
%   - H:        Current approximation of the Hessian (matrix)
%   - H_posdef: Indicates whether H is positive definite or not (boolean)
%   - grad:     Current iterates gradient (vector)
%   - delta:    Current trust region radius (positive scalar)
%
% Outputs:
%   - s:                   Dogleg step, i.e., an inexact solution of the
%                          simple (non-constrained) trust region subproblem
%                          (vector)
%   - predicted_reduction: Predicted reduction corresponding to the
%                          dogleg step s (scalar)

% Choose a mass matrix
M = eye(size(H)); % Corresponds to evenly distributed mass

% If the approximation of the Hessian is positiv definite, calculate the
% Newton step
if H_posdef > 0
    warnID = 'MATLAB:nearlySingularMatrix';
    warning('off', warnID);
    s_newton = - H \ (M * grad);
    norm_s_newton = sqrt(s_newton' * M * s_newton);

    % If the Newton step is feasible, choose it to be the dogleg step
    % direction, since it is the global solution to the subproblem
    if norm_s_newton <= delta
        s = s_newton;
        
    else % If the Newton point is not feasible, first calculate the Cauchy point and its norm
        norm_grad = sqrt(grad' * M * grad);
        denom_scalprod = grad' * H * grad;
        if denom_scalprod <= 0
            tau = 1;
        else
            tau = min(1 , norm_grad^3 / (delta * denom_scalprod));
        end
        s_cauchy = - tau * delta * (grad / norm_grad);
        norm_s_cauchy = sqrt(s_cauchy' * M * s_cauchy);
        
        % Calculate a feasible convex combination of the Newton
        % step and the Cauchy point by calculating the coefficients
        % of the quadratic equation |s_dogleg|^2 = delta^2
        a_coeff = 0.5 * (s_newton - s_cauchy)' * M ...
            * (s_newton - s_cauchy);
        b_coeff = s_cauchy' * M * (s_newton - s_cauchy);
        c_coeff = 0.5 * (norm_s_cauchy^2 - delta^2);

        % Numerically stable method for solving the 1-D quadratic
        % (taken from Matlab's build-in dogleg method)
        q_coeff = -0.5 * (b_coeff + sign(b_coeff) ...
            * sqrt(b_coeff^2 - 4 * a_coeff * c_coeff));
        if b_coeff > 0
            t = c_coeff / q_coeff;
        else
            t = q_coeff / a_coeff;
        end

        % Assembling the dogleg step and computing its norm
        s = s_cauchy + t * (s_newton - s_cauchy);
        norm_s = sqrt(s' * M * s);

        % Check whether the dogled step is feasible and throw a
        % warning, if this is not the case
        if (abs(norm_s - delta) > 1e-10)
            warning('There is a problem with the Dogleg-step!');
        end
    end
else % If H is not positive definite, simply calculate the Cauchy point
    norm_grad = sqrt(grad' * M * grad);
    denom_scalprod = grad' * H * grad;
    if denom_scalprod <= 0
        tau = 1;
    else
        tau = min(1 , norm_grad^3 / (delta * denom_scalprod));
    end
    s = - tau * delta * (grad / norm_grad);
end

% Calculate the predicted reduction of the objective function that is
% realized by the dogleg step
predicted_reduction = - (grad' * M * s + 0.5 * s' * H * s);
end