clear; clc; close all;

%% Randomly choose or load a test instance of the Transportation Identification Problem (TIP)
% scenario = "random";
scenario = "test_50x50";

% If no data is specified, choose random marginals
if isequal(scenario, "random")
    % Define model parameters and create cost matrix
    n1 = 20; n2 = 20;
    p = 2;
    cost = @(i,j) abs(j-i).^p;
    [X, Y] = meshgrid(1:n1, 1:n2);
    c_d = cost(X, Y)';

    % Choose (sparse) random marginals
    % rng(1337);
    sparse_marg = false; % Setting sparse_marg to 1 enforces sparsity of the marginals
    if sparse_marg == true
        sparsity = 0.75;
        mu1_d = sprand(n1, 1, 1 - sparsity);
        mu1_d = mu1_d / sum(mu1_d);
        mu2_d = sprand(n2, 1, 1 - sparsity);
        mu2_d = mu2_d / sum(mu2_d);
    else
        mu1_d = rand(n1, 1);
        mu1_d = mu1_d / sum(mu1_d);
        mu2_d = rand(n2, 1);
        mu2_d = mu2_d /sum(mu2_d);
    end

    % Calculate optimal transport plan between mu1_d and mu2_d using
    % MATLAB's linprog
    if sparse_marg == true
        pi_d = sparse(solve_1dkantorovich(mu1_d, mu2_d, c_d));
    else
        pi_d = solve_1dkantorovich(mu1_d, mu2_d, c_d);
    end
    pi_d = max(0,pi_d);

    % Construct initial data
    mu1_0 = 1/n1 * ones(size(mu1_d));
    H_0 = eye(n1);
    delta_0 = 1;
    
else % Otherwise, load the scenario specified above
    load(scenario);
end

%% Choose observation domains
n1 = numel(mu1_d); n2 = numel(mu2_d);
whole_domain = 1;
if whole_domain % TIP on the whole domain
    D1 = ones(n1, 1);
    D = ones(n1, n2);
else
    shape = "stripe_shift";
    if isequal(shape, "stripe") % TIP on diagonal stripe
        width = 3;
        D1 = zeros(n1, 1);
        D1(max(1, floor(n1/2) - width):min(floor(n1/2) + width, n1)) = 1;
        D1(33:50) = 1;
        D = zeros(n1, n2);
        for i = 1:n1
            D(i, max(1, i - width):min(i + width, n2)) = 1;
        end
    elseif isequal(shape, "stripe_shift") % TIP on shifted diagonal stripe
        D1 = zeros(n1, 1);
        D1(33:50) = 1;
        D = zeros(n1, n2);
        for i = 1:n1
            D(min(i+8, n1), max(1, i - width):min(i + width, n2)) = 1;
        end
        % D(50,46:50) = 0;
        D(1:7,1) = 1;
        D(8,1:3) = 1;
        D(7,1:2) = 1;
        D(6,1:1) = 1;
    elseif isequal(shape, "holes") % TIP on domain with holes
        k = 2;
        D1 = zeros(n1,1);
        D1(ind_sets{k}) = 1;
        D = zeros(n1, n2);
        D(ind_sets{k}, :) = 1;
        D(:, setdiff(1:n1, ind_sets{k})) = 0;
    end
end

%% Construct the target function and its (relevant) gradients
% Weighting parameter
lambda = 1e-0;

% Target function
if issparse(pi_d)
    % Use full() to convert 1x1 sparse into double
    J = @(pi, mu1) full( ...
        1/2 * sum((D .* (pi - pi_d)).^2, "all") + lambda/2 * sum((D1 .* (mu1 - mu1_d)).^2) ...
        );
else
    J = @(pi, mu1) 1/2 * sum((D .* (pi - pi_d)).^2, "all") + lambda/2 * sum((D1 .* (mu1 - mu1_d)).^2);
end

% Gradient of target function
grad_pi_J = @(pi) D .* (pi - pi_d);
grad_mu1_J = @(mu1) lambda * D1 .* (mu1 - mu1_d);

%% Collect the data to pass to the algorithm
% Regularization parameters (for gamma*epsilon <= 1e-08 the reg. dual
% solver becomes unstable!)
gamma_0 = 1e-3;
epsilon_0 = gamma_0;

% Target function data
target_data.J = J;
target_data.grad_pi_J = grad_pi_J;
target_data.grad_mu1_J = grad_mu1_J;

% Model data
model_data.n1 = numel(mu1_d);
model_data.n2 = numel(mu2_d);
model_data.mu2_d = mu2_d;
model_data.c_d = c_d;

% Initial data
initial_data.mu1_0 = mu1_0;
initial_data.delta_0 = delta_0;
initial_data.H_0 = H_0;
initial_data.epsilon_0 = epsilon_0;
initial_data.gamma_0 = gamma_0;

% Termination data
termination_data.max_iter = 200;
termination_data.stat_TOL = 1e-4; %1e-4 or 1e-6

% Additional data
additional_data.diagnostics = 1;
additional_data.check_definiteness = 1;
additional_data.solver_TOL = 1e-8; %1e-8 or 1e-10
additional_data.set_TOL = 1e-10;
additional_data.max_subgrads = 1*(n1+n2); %1*(n1+n2) or 10*(n1+n2)

%% Calling the non-smooth trust region method
[mu1, pi, output] = cntr_method(target_data, model_data, initial_data, ...
    termination_data, [], additional_data);

%% Plotting the sparsity patterns of observations and computed variables
plot_sparsity = 1;
if plot_sparsity
    figure
    hold on
    % Plot the domain, the true optimal transport plan, and the computed
    % plan
    if ~whole_domain
        spy(D, "square"); % "+"/"square"
    end
    spy(pi_d, "square");
    spy(pi, "square");

    % Customize the markers
    lightgray = [0.9, 0.9, 0.9];
    lines = findobj(gcf, 'Type', 'Line');
    lines(1).MarkerFaceColor = "cyan";
    lines(1).MarkerEdgeColor = "cyan";
    lines(1).MarkerSize = 8;
    lines(2).MarkerFaceColor = "red";
    lines(2).MarkerEdgeColor = "red";
    lines(2).MarkerSize = 10;
    
    if whole_domain
        legend(strcat("Sparsity pattern of ", '$\pi_d$'), ...
            strcat("Sparsity pattern of ", '$\bar{\pi}$'), ...
            'Interpreter', 'latex', 'Location', 'southwest');
    else
        lines(3).MarkerEdgeColor = lightgray;
        lines(3).MarkerFaceColor = lightgray;
        lines(3).MarkerSize = 9;
        legend("Domain $D$", strcat("Sparsity pattern of ", '$\pi_d$'), ...
            strcat("Sparsity pattern of ", '$\bar{\pi}$'), 'Interpreter', 'latex');
    end
    set(gca, "TickLength", [0.025, 0.05]);
    set(gca, "FontSize", 14);
    xlabel("\Omega_2", "FontSize", 16); ylabel("\Omega_1", "Rotation", 0, "FontSize", 16);

    figure
    hold on
    if ~whole_domain
        % Plot the domain D1
        bar(D1, "LineStyle", "none", "FaceColor", lightgray, "BarWidth", 1);
    end
    xlim([0.5, n1 + 0.5]); %ylim([0, round(1.5*max([mu1; mu1_d]), 2)]);
    ylim([0, 0.5]);

    % Plot the origin marginal and the accumulation point of the trust
    % region method
    bar(mu1_d, "r", "BarWidth", 1);
    bar(mu1, "c", "BarWidth", 0.6);
    if whole_domain
        legend("$\mu_1^d$", "$\bar{\mu}_1$", "Interpreter", "latex");
    else
        legend("Domain $D_1$", "$\mu_1^d$", "$\bar{\mu}_1$", "Interpreter", "latex");
    end
    set(gca, "TickLength", [0.025, 0.05]);
    set(gca, "FontSize", 14);
    xlabel("\Omega_1", "FontSize", 16); ylabel("Marginal value", "FontSize", 16);
end

% Plot stationartiy plots
plot_stationarity = 0;
if plot_stationarity
    stat_log = output.stat_log;
    semilogy(stat_log);
    xlabel("Iteration"); ylabel("\theta_R / \psi_R", "Rotation", 0);
    set(gca, "TickLength", [0.025, 0.05]);
    set(gca, "FontSize", 14);
end

% Plot target function plot
plot_target = 0;
if plot_target
    target_log = output.target_log;
    semilogy(target_log);
    set(gca, "TickLength", [0.025, 0.05]);
    set(gca, "FontSize", 14);
    xlabel("Iteration"); ylabel("Target function value");
end

%% Auxiliary code
% lines = findobj(gcf,'Type','Line');
% for i = 1:numel(lines)
%   lines(i).LineWidth = 1.25;
% end