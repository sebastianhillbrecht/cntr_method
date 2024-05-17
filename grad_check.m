function grad_check(J, g, mu1, mu2, c, gamma, epsilon, solver_TOL)
%GRAD_CHECK A method to verify the calculation of gradients of the reduced
%target function (w.r.t. mu1) by computing the difference quotient for
%different stepsizes; if the gradient was calculated correctly, the loglog
%plot output by the method should show a V-shape
%
% Inputs:
%   - J: Target function (function handle)
%   - g: Candidate for the gradient (vector)
%   - mu1: Given source marginal (vector)
%   - mu2: Given target marginal (vector)
%   - c: Given cost matrix (matrix)
%   - gamma: Regularization parameter for Kantorovich problem (positive scalar)
%   - epsilon: Regularization parameter for dual problem (positive scalar)
%   - solver_TOL: Tolerance for the control-to-state solver (positive scalar)

% Choose maximum (initial) step size and number of discretization points
H = 1;
N = 50;

% Initialization
n1 = numel(g); 
fin_grad = zeros(n1,1);
norm_diff = zeros(N, 1);

% Compute the solution pi corresponding to given point (mu1, mu2)
[~, ~, iter, pi, residual] = solve_reg_dual(mu1, mu2, c, gamma, epsilon, solver_TOL);

% 
for i = 1:N
    h = H / (2^(i-1));
    
    % Compute every partial difference quotient
    for j = 1:n1
        mu_h = mu1; mu_h(j) = mu_h(j) + h;
        [~, ~, iter_h, pi_h, residual_h] = solve_reg_dual(mu_h, mu2, c, gamma, epsilon, solver_TOL);
        fin_grad(j) = (J(pi_h, mu_h) - J(pi, mu1)) / h;
    end
    norm_diff(i) = norm(g - fin_grad);
end

% Construct log-log plot with the computed data
loglog(h./2.^(1:N),norm_diff);