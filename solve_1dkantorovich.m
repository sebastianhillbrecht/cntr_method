function pi = solve_1dkantorovich(mu1, mu2, cost)
%SOLVE_1DKANTOROVICH Solves the Kantorovich problem in the case that the
%marginals are vectors using Matlab's linear problem solver
%
% Inputs:
%   - mu1:  Source marginal (vector)
%   - mu2:  Target marginal (vector)
%   - cost: Cost matrix (matrix)
%
% Outputs:
%   - pi: Optimal transport plan between mu1 and mu2 w.r.t. cost (matrix)

if ~ isvector(mu1)
    error('mu1 must be a vector!')
elseif ~ isvector(mu2)
    error('mu2 must be a vector!')
end

if abs(sum(mu1) - sum(mu2)) > 100 * eps
    error('mu1 and mu2 must have same mass!')
end

m = numel(mu1); n = numel(mu2);

% Reshape mu1 and mu2 into column vectors
mu1 = mu1(:); mu2 = mu2(:);

if numel(cost) == 1 && cost >= 1
    alpha = cost;
    c = @(i,j) abs(i - j) .^ alpha;
    [I, J] = ndgrid(1:m, 1:n);
    cost = c(I, J);
elseif any(size(cost) ~= [m, n])
    error('cost must be suitable matrix or value >= 1!')
end

% Construct the system matrix describing the vectorized optimal transport
% problem
A = [repmat(eye(m), 1, n); repmat(zeros(n, m), 1, n)];
for i = 1:n
    A(m+i, ((i-1)*m+1):(i*m)) = 1;
end

% Right-hand side vector
b = [mu1; mu2];

% Cost function as a vector
c = reshape(cost, n*m, 1);

% Solve optimal transport problem using Matlab's linprog
options = optimoptions("linprog", "Display", "none");
[pi_vec, ~, exitflag, output, lambda] = linprog(c, -eye(m*n), zeros(m*n,1), A, b, [], [], options);

% Check solution, give warning
if exitflag ~= 1
    warning('Check output of linprog! Transport plan might not be optimal.')
end

% Reshape solution to matrix
pi = reshape(pi_vec, m, n);

% Extract dual variables
alpha = lambda.eqlin(1:m);
beta = lambda.eqlin(m+1:m+n);