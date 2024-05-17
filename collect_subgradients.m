function [subgrads, Nsubgrads] = collect_subgradients(grad_pi_J, grad_mu1_J, ...
    delta, c, mu1_0, mu2_0, gamma, epsilon, set_TOL, solver_TOL, max_subgrads)
%COLLECT_SUBGRADIENTS WARNING: This method is highly inefficient and
%likely to fail in dimensions n1,n2 > 50!
%
%Collects a certain number of elements of the collective Bouligand
%subdifferential by iteratively choosing different points from the ball
%around the current iterate and fetching subgradients at those points
%
% Inputs:
%   - grad_pi_J:    Target function's gradient w.r.t. the state pi (function handle)
%   - grad_mu1_J:   Target function's gradient w.r.t. the control mu1 (function handle)
%   - delta:        Current trust region radius (positive scalar)
%   - c:            Cost matrix (matrix)
%   - mu1_0:        Current source marginal (vector)
%   - mu2_0:        Current target marginal (vector)
%   - gamma:        Kantorovich regularization parameter (positive scalar)
%   - epsilon:      Dual problem regularization parameter (positive scalar)
%   - set_TOL:      Tolerance to decide which indices shall be considered
%                   to be active/biactive (positive scalar)
%   - solver_TOL:   Optimality tolerance for the solver of the
%                   control-to-state mapping (positive scalar)
%   - max_subgrads: Desired number of subgradients that shall be computed
%                   (positive integer)
% 
% Outputs:
%   - subgrads:  List of subgradients (matrix)
%   - Nsubgrads: Number of computed subgradients (positive integer)

% Fetching dimensions
n1 = numel(mu1_0);
n2 = numel(mu2_0);

% Initialization
num_subgrads = 0;
done = 0;
it = 0;

% Iteration
while (it <= 2 * (n1 + n2)) && (~done)
    % Choosing different points from the ball around (mu1_0, mu2_0) by,
    % in every iteration, choosing a (n1+n2)-dimensional unit vector and
    % going once in positive and once in negative direction
    if it == 0
        mu1 = mu1_0;
        mu2 = mu2_0;
    elseif it <= 2 * n1
        v1 = zeros(n1, 1);
        v1(ceil(it/2)) = (-1)^(mod(it,2));
        mu1 = mu1_0 + delta * v1;
        mu2 = mu2_0;
    else
        v2 = zeros(n2, 1);
        v2(ceil((it-2*n1)/2)) = (-1)^(mod(it,2));
        mu1 = mu1_0;
        mu2 = mu2_0 + delta * v2;
    end

    % For every point (mu1, mu2), calculate the corresponding state pi and
    % the dual variables
    [alpha, beta, k, pi, residual] = solve_reg_dual(mu1, mu2, c, ...
        gamma, epsilon, solver_TOL);

    % Determine the matrices corresponding to the sets Omega+(mu1, mu2) and
    % Omega0(mu1, mu2)
    OmegaPlusMat = (alpha + beta' - c) > set_TOL;
    OmegaZeroMat = abs(alpha + beta' - c) < set_TOL;

    % Compute the subgradients corresponding to the points (mu1, mu2);
    % for every point, choose the subset A to be the empty set
    [sgs, num, k] = compute_subgradients(OmegaPlusMat, OmegaZeroMat, ...
        grad_pi_J(pi), grad_mu1_J(mu1), epsilon, gamma, max_subgrads);
    collect(:,(num_subgrads+1):(num_subgrads+num)) = sgs(:,1:num);

    % Remove subgradients that are identical or too close to each other
    unique_TOL = 1e-5; % This tolerance depends on the problem structure
    collect = uniquetol(collect', unique_TOL, 'ByRows', true)';
    num_subgrads = size(collect, 2);

    % Break, if enough subgradients have been computed
    if num_subgrads >= max_subgrads
        done = 1;
    end
    it = it+1;
end

% Count subgradients and return the desired amount, if possible
Nsubgrads = min(max_subgrads, num_subgrads);
subgrads = collect(:, 1:Nsubgrads);