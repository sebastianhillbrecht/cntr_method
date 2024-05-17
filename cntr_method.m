function [mu1, pi, output] = cntr_method(target_data, model_data, ...
    initial_data, termination_data, method_parameters, additional_data)
%CNTR_METHOD This function is an implementation of the constrained
%nonsmooth trust region method (CNTR method) applied to the transportation
%identification problem (TIP).
%
% Inputs:
%   - target_data:        A struct with the fields (each is a function handle)
%                            o 'J':          target function;
%                            o 'grad_pi_J':  target function's gradient w.r.t. the state pi;
%                            o 'grad_mu1_J': target function's gradient w.r.t. the control mu1.
%
%   - model_data:         A struct that contains information on the control-to-state model, i.e., in the
%                         case of TIP, it contains the fields
%                            o 'n1':    dimension of source marginal (positive integer);
%                            o 'n2':    dimension of target marginal (positive integer);
%                            o 'mu2_d': fixed target marginal (vector);
%                            o 'c_d':   fixed target cost matrix (matrix).
%
%   - initial_data:       A struct with the fields
%                            o 'mu1_0':     initial control (vector);
%                            o 'delta_0':   initial trust region radius (positive scalar);
%                            o 'H_0':       initial approximation of the Hessian (matrix);
%                            o 'gamma_0':   initial Kantorovich regularization parameter (positive scalar);
%                            o 'epsilon_0': initial dual regularization parameter (positive scalar).
%
%   - termination_data:   A struct with the fields
%     (optional)             o 'max_iter:  maximum number of iterations (positive integer);
%                            o 'stat_TOL': stationarity tolerance (positive scalar).
%
%   - methods_parameters: A struct with the fields (each is a positive scalar)
%     (optional)             o 'R':         radius for computation of stationarity measures;
%                            o 'delta_min': trust region radius threshold for complex subproblem;
%                            o 'delta_max': maximum trust region radius;
%                            o 'eta_1':     quality indicator threshold for unsuccessful step;
%                            o 'eta_2':     quality indicator threshold for very successful step;
%                            o 'beta_1':    trust region shrinkage parameter;
%                            o 'beta_2':    trust region enlargement parameter;
%                            o 'nu':        parameter for generalized Cauchy decrease condition.
%
%   - additional_data:    A struct with the fields
%     (optional)             o 'diagnostics':        enable/disable diagnostic command window output (boolean);
%                            o 'check_definiteness': enable/disable regularity check of Hessian approximation (boolean);
%                            o 'solver_TOL':         control-to-state solver tolerance (positive scalar);
%                            o 'set_TOL':            tolerance for assignment of indices to active/biactive/inactive set (positive scalar);
%                            o 'max_subgrads':       maximum number of subgradients for complex trust region subproblem (positive integer).
%
%   Outputs:
%       - mu1:    The final iterate, i.e., the control mu1 for which the CNTR method terminated (vector).
%
%       - pi:     The state corresponding to the final iterate (matrix).
%
%       - output: A struct with the fields
%                   o 'flag':             description of why the method was terminated (string);
%                   o 'stationarity':     value of stationarity measure corresponding to final iterate (nonnegative scalar);
%                   o 'target_val':       value of target function at final iterate (scalar);
%                   o 'delta':            final trust region radius (positive scalar);
%                   o 'k':                final iteration number (positive integer);
%                   o 'successful_steps': number of successful steps (positive integer);
%                   o 'complex_count':    number of iterations within complex trust region subproblem (positive integer);
%                   o 'alpha':            first dual variable corresponding to final iterate (vector);
%                   o 'beta':             second dual variable corresponding to final iterate (vector);
%                   o 'stat_log':         log of stationarity measures (vector of nonnegative scalars);
%                   o 'target_log':       log of target function values (vector of nonnegative scalars).
%

%% Extraction of input data % % % % % % % % % % % % % % % % % % % % % % % %
% Extract the target function's data
J = target_data.J;
grad_pi_J = target_data.grad_pi_J;
grad_mu1_J = target_data.grad_mu1_J;

% Extracting the control-to-state model's data
n1 = model_data.n1;
n2 = model_data.n2;
mu2_d = model_data.mu2_d;
c_d = model_data.c_d;

% Extract the initial values
mu1_0 = initial_data.mu1_0;
delta_0 = initial_data.delta_0;
H_0 = initial_data.H_0;
gamma_0 = initial_data.gamma_0;
epsilon_0 = initial_data.epsilon_0;

% Extracting the termination parameters (if given)
if exist('termination_data', 'var') && isa(termination_data, 'struct')
    max_iter = termination_data.max_iter;
    stat_TOL = termination_data.stat_TOL;
else
    % Use the default parameters
    max_iter = 200;
    stat_TOL = 1e-6;
end

% Extracting the trust region method's parameters (if given)
if exist('method_parameters', 'var') && isa(method_parameters, 'struct')
    % Trust region method
    R = method_parameters.R;
    delta_min = method_parameters.delta_min;
    delta_max = method_parameters.delta_max;
    eta_1 = method_parameters.eta_1;
    eta_2 = method_parameters.eta_2;
    beta_1 = method_parameters.beta_1;
    beta_2 = method_parameters.beta_2;
    nu = method_parameters.nu;

    % Path following heuristic
    path_following = method_parameters.path_following;
    path_shrinkage = method_parameters.path_shrinkage;
    path_frequency = method_parameters.path_frequency;
    gamma_min = method_parameters.gamma_min;
    epsilon_min = method_parameters.epsilon_min;
else
    % Use the default parameters
    % Trust region method
    R = sqrt(n1);
    delta_min = 1e-6;
    delta_max = sqrt(n1);
    eta_1 = 0.1;
    eta_2 = 0.9;
    beta_1 = 0.5;
    beta_2 = 1.5;
    nu = 1;

    % Path following heuristic
    path_following = 0;
    path_shrinkage = 0.1;
    path_frequency = 5;
    gamma_min = 1e-6;
    epsilon_min = 1e-6;
end

% Extracting the additional data (if given)
if exist('additional_data', 'var') && isa(additional_data, 'struct')
    diagnostics = additional_data.diagnostics;
    if diagnostics
        yesno{1} = 'Yes'; yesno{2} = 'No';
    end
    check_definiteness = additional_data.check_definiteness;
    solver_TOL = additional_data.solver_TOL;
    set_TOL = additional_data.set_TOL;
    max_subgrads = additional_data.max_subgrads;
else
    % Use the default parameters
    diagnostics = 1; yesno{1} = 'Yes'; yesno{2} = 'No';
    check_definiteness = 1;
    solver_TOL = 1e-10;
    set_TOL = 1e-10;
    max_subgrads = 2*n1*n2;
end

%% Initialization % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Mute linprog
options = optimoptions("linprog", "Display", "none");

% Initialization of trust region method
k = 0; resolve = 0; done = 0;
nullstep = 0; null_count = 0; complex_count = 0;
mu1 = mu1_0; delta = delta_0; H = H_0;
gamma = gamma_0; epsilon = epsilon_0;
stat_log = zeros(1, max_iter); target_log = zeros(1, max_iter);

%% Constrained nonsmooth trust region method % % % % % % % % % % % % % % %
while ~done
    % Print diagnosticts during the iteration (if desired)
    if diagnostics && mod(k, 10) == 0
        fprintf(['Iter. | Target val. | Stat. meas. | gamma | epsi. | TR radius | Norm step | ' , ...
            'Qual. ind. | H pd.? | Cmplx? | Nulls.? | Feas.?\n']);
    end

    %% Calculation of current state, gradient, and Hessian approximation
    % In case of a nullstep, skip the calculation of state pi, gradient g
    % and Hessian H, since only the trust region radius delta is modified;
    % if path following is enabled and a regularization parameter was
    % changed, recalculate all variables
    if ~nullstep || resolve
        %% Calculate the state corresponding to the current iterate
        if (k == 0) || resolve
            % Calculate the state, i.e., the solution to the regularized
            % dual problem of the regularized Kantorovich problem
            [alpha, beta, iter, pi, residual] = solve_reg_dual(mu1, mu2_d, c_d, ...
                gamma, epsilon, solver_TOL); % For gamma*epsilon <= 1e-08 this method becomes unreliable!
            J1 = J(pi, mu1);

            % Print initial target function value 
            if diagnostics && (k == 0)
                fprintf([ ' 0      %-10.5g    -             -       -       -           -' , ...
                    '           -            -        -         -' , ...
                    '         -\n'], J1);
            end

            % Reset resolve
            resolve = 0;
        else
            % Accept the state pi, the dual variables alpha & beta, and the
            % target function value from the calculation of the actual
            % reduction in the previous step
            pi = pi1;
            alpha = alpha1;
            beta = beta1;
            J1 = J2;
        end
        
        %% Calculate a gradient of the reduced target function for the current iterate
        % Determine the matrices corresponding to the sets Omega+_k and
        % Omega0_k
        OmegaPlusMat = (alpha + beta' - c_d) > set_TOL;
        OmegaZeroMat = abs(alpha + beta' - c_d) < set_TOL;

        % Choose some subset A of Omega0_k and construct the corresponding
        % characteristic matrix (my implementation leaves room for
        % improvement!)
        AMat = zeros(n1, n2); % Correspond to A = \empty
        % AMat = Omega0Mat; % Corresponds to A = Omega0_k

        % Calculate the subgradient g_k (corresponding to the set A) of the
        % reduced target function
        OmegaPlusAMat = OmegaPlusMat + AMat; % Corresponds to Omega+_k \cup A
        cropMat = OmegaPlusAMat .* grad_pi_J(pi); % Corresponds to masking w.r.t. Omega+_k \cup A
        systemMat = [diag(OmegaPlusAMat * ones(n2,1)), OmegaPlusAMat;
                     OmegaPlusAMat',                   diag(OmegaPlusAMat' * ones(n1,1))];
        uv = [sum(cropMat,2); sum(cropMat,1)'];
        gh = (systemMat + gamma * epsilon * eye(n1+n2)) \ uv;
        g = gh(1:n1) + grad_mu1_J(mu1);
        % h = gh(n1+1:n1+n2); % Gradient w.r.t. the target marginal (not needed!)

        % % Check, whether the computed gradient matches its finite
        % % difference approximation: the plot should show a V-shape
        % grad_check(J, g, mu1, mu2_d, c_d, gamma, epsilon, solver_TOL)

        %% Calculate an approximation of the Hessian of the reduced target function for the current iterate
        % At every 10th iteration or if the approximation's norm exceeds
        % some threshold, reset the approximation
        if  mod(k - null_count, 10) == 0 || norm_H > 10
            H = H_0;
        else
            % BFGS-update formula for a symmetric, positive definite
            % approximation of the Hessian
            g_diff = g - g_old;
            H = H - ((H * d) * (H * d)') / (d' * H * d) + ...
                (g_diff * g_diff') / (g_diff' * d);
        end

        % Calculate the approximation's Frobenius norm for the Cauchy
        % decrease condition
        norm_H = norm(H, 'fro');

        % If desired, test for H's definiteness via using the Cholesky
        % factorization implemented in Matlab
        if check_definiteness
            [~, r] = chol(H);
            if r == 0 && isequal(H, H') && all(diag(H) > set_TOL)
                % If r == 0, then H is at least positive semidefinite; if,
                % additionally, all entries on the main diagonal are
                % positive, then H is positive definite
                H_posdef = 1;
            else
                H_posdef = 0;
            end
        else
            % If check_definiteness is set to 0, we automatically assume
            % that H is pd (for simplicity; however, this may lead to
            % problems with the dogleg step calculation!)
            H_posdef = 1;
        end
    end

    %% Compute the stationarity measure for the current iterate and the current gradient
    % Because we set R = sqrt(m) and because of the structure of the
    % standard simplex, the calculation of the stationarity measure reduces
    % to solving an LP
    [d_theta, val] = linprog(g, [], [], ones(1, n1), 0, -mu1, 1 - mu1, options);
    theta = - val;

    % Check, whether the stationarity measure at g_k is close to 0,
    % i.e., wether we reached a first-order optimal point; in this case,
    % terminate the method and fetch some output data
    if theta < stat_TOL
        done = 1;
        flag = ['Termination: Stationarity measure is less than stationarity tolerance ', ...
            num2str(stat_TOL), '.'];
        stationarity = [num2str(theta), ' (regular)'];
        stat_log(k+1) = theta;
        
    else % Otherwise, continue with the iteration
        % If the trust region radius is large, solve the simple trust
        % region subproblem
        if delta >= delta_min
            %% Simple trust region subproblem
            % Compute an inexact solution of the trust region subproblem
            % using an optimal convex combination of the projected dogleg
            % point and the linearized solution from the calulation of the
            % stationarity measure
            d_dl = dogleg(H, H_posdef, g, delta); % Dogleg point
            proj_dl = projsplx(d_dl + mu1) - mu1; % Projection of dogleg point onto standard simplex
            tau = solve_quadratic(proj_dl, d_theta, H, g, delta);
            d = proj_dl + tau * (d_theta - proj_dl); % Optimal convex combination (tau is typically close to 0!)

            % Calculate the predicted reduction of the convex combination
            predicted_red = - (g' * d + 0.5 * d' * H * d);

            % Check the generalized Cauchy decrease condition
            rhs = nu / (2*R) * theta * min([R, delta, theta / (R * norm_H)]);
            if predicted_red < rhs
                % If the generalized Cauchy decrease condition is not
                % satisfied, compute a save guard step that always
                % satisfies it (cf. blueprint paper/my thesis!)
                if theta >= norm(d_theta)^2 * norm_H * min(1, delta/R)
                    scal = min(1, delta/R);
                else
                    scal = theta / (norm(d_theta)^2 * norm_H);
                end
                d = scal * d_theta;

                % Recalculate the predicted reduction w.r.t. the save guard
                % step
                predicted_red = - (g' * d + 0.5 * d' * H * d);
            end

            % Calculate the step's norm
            norm_d = sqrt(d' * d);

            % Store the current iterate for later
            mu1_old = mu1;

            % Compute the (supposedly) next iterate as well as the
            % corresponding state and dual variables
            mu1 = mu1 + d;
            [alpha1, beta1, iter, pi1, residual] = solve_reg_dual(mu1, mu2_d, c_d, ...
                gamma, epsilon, solver_TOL);

            % Compute the quality indicator rho
            J2 = J(pi1, mu1);
            actual_red = J1 - J2;
            rho = actual_red / predicted_red;

            % Save the stationarity measure for diagnostic purposes
            stat_log(k+1) = theta;
            
        else % If the trust region radius is small, solve the complex trust region subproblem
            %% Complex trust region subproblem
            % Compute an approximation of the collective Bouligand
            % subdifferential containing up to max_subgrads subgradients
            [subgrads, Nsubgrads] = collect_subgradients(grad_pi_J, grad_mu1_J, delta, ...
                c_d, mu1, mu2_d, gamma, epsilon, set_TOL, solver_TOL, max_subgrads);

            % Compute an approximation of the complex stationarity measure
            % using the above approximation of the collective Bouligand
            % subdifferential and Matlab's linprog
            [dxi, val, ~, ~] = linprog([zeros(n1,1); 1], ...
                [subgrads', -ones(Nsubgrads,1)], zeros(Nsubgrads,1), ...
                [ones(1,n1), 0], 0, [-mu1; - Inf], [1 - mu1; Inf], options);
            d_statmeas = dxi(1:n1);
            psi = -val;

            % Check whether the complex stationarity measure is close to 0,
            % i.e., wether we reached a first-order optimal point; in this
            % case, terminate the method and fetch some output data
            if psi <= stat_TOL
                flag = ['Termination: Stationarity measure less than ', ...
                    num2str(stat_TOL), '.'];
                stationarity = [num2str(psi), ' (complex)'];
                stat_log(k+1) = psi;
                break
            end

            % Compute a decent step that always satisfies the complex
            % generalized Cauchy decrease condition (cf. blueprint paper/my
            % thesis!) and calculate its norm and the predicted reduction
            if psi >= norm(d_statmeas)^2 * norm_H * min([1, delta/R])
                % scal = min([1, delta/R]);
                scal = min([1, delta]) / norm(d_statmeas);
            else
                scal = psi * norm(d_statmeas)^2 / norm_H;
            end
            d = scal * d_statmeas;
            norm_d = sqrt(d' * d);
            predicted_red = scal * psi - 0.5 * d' * H * d;

            % To be sure, check the feasibility of d and the complex
            % generalized Cauchy decrease condition
            if sum(d) > 100 * eps || any(d <= -mu1 - 100*eps) || ...
                    any(d >= (1 - mu1) + 100*eps)
                warning('Step in complex case is not feasible!');
            end
            rhs = nu / (2*R) * psi * min([R, delta, psi / (R * norm_H)]);
            if predicted_red < rhs
                warning(['Generalized Cauchy decrease condition not' ...
                    'satisfied in the complex case!'])
            end

            % Store the current iterate in dummy variable
            mu1_old = mu1;
            
            % Compute the (supposedly) next iterate as well as the
            % corresponding state and dual variables
            mu1 = mu1 + d;
            [alpha1, beta1, ~, pi1] = solve_reg_dual(mu1, mu2_d, c_d, ...
                gamma, epsilon, solver_TOL);

            % Compute the quality indicator of the complex model
            J2 = J(pi1, mu1);
            actual_red = J1 - J2;
            if psi > theta * delta
                rho = actual_red / predicted_red;
            else
                rho = 0;
            end

            % Increase the complex case counter and save the stationarity
            % measure for diagnostics
            complex_count = complex_count + 1;
            stat_log(k+1) = psi;
        end
        %% Update step % % % % % % %
        k = k+1;
        g_old = g;

        % Update rule for the control variable
        if rho < eta_1
            nullstep = 1;
            null_count = null_count + 1;
            mu1 = mu1_old;
        else
            nullstep = 0;
        end

        % Save target function value
        target_log(k) = nullstep * J1 + (~nullstep) * J2;

        % If desired, print some iteration information
        if diagnostics
            % Target value
            fval = target_log(k);

            % Check feasibility of control variable
            mu_feas = all(mu1 >= - 100 * eps) && abs(sum(mu1) - 1) < 100 * eps;
            fprintf([' %-4d   %-10.5g    %-10.5g    %-6.0g  %-6.0g  %-8.3g    %-8.3g    %-9.3g' , ...
                '    %-5s    %-6s    %-6s    %-5s\n'], k, target_log(k), stat_log(k), ...
                gamma, epsilon, delta, norm_d, rho, yesno{2 - H_posdef}, ...
                yesno{2 - (delta < delta_min)}, yesno{2 - nullstep}, yesno{2-mu_feas});
        end

        % Update the trust region radius
        if ~done
            if rho <= eta_1
                delta = beta_1 * delta;
            elseif rho > eta_2
                delta = min(max(delta_min, beta_2 * delta), delta_max);
            else
                delta = min(max(delta_min, delta), delta_max);
            end
        end

        % Check whether the iteration counter is greater than max_iter
        if k >= max_iter
            done = 1;
            flag = ['Abortion: Iteration counter greater than ', num2str(max_iter), '!'];
            stationarity = [num2str(theta), ' (regular)'];
            stat_log(k+1) = theta;
            target_log(k)
        end

        % Path following routine to reduce the effects of the
        % regularizations
        if path_following && (mod(k, path_frequency) == 0)
            % Compute the violation of the primal problem's and the dual
            % problem's constraints
            primal_violation = max(abs([pi * ones(n2,1); pi' * ones(n1,1)] - [mu1; mu2_d]));
            dual_violation = max(alpha + beta' - c_d, [], "all");

            % Decide which regularization parameter shall be shrinked
            if (dual_violation >= primal_violation) && (gamma >= 100 * epsilon)
                % gamma penalizes the dual constraint alpha + beta' <= cost
                gamma = max(gamma * path_shrinkage, gamma_min);

                % If gamma does not change, do not resolve as this would
                % cause problems with the Hessian approximation
                resolve = ~(gamma == gamma_min);
            else
                % epsilon penalizes the primal constraints pi*one = mu & pi'*one = nu
                epsilon = max(epsilon * path_shrinkage, epsilon_min);

                % If epsilon does not change, do not resolve as this would
                % cause problems with the Hessian approximation
                resolve = ~(epsilon == epsilon_min);
            end
        end
    end
end


%% Gather some output information % % % % % % %
output.flag = flag;
output.stationarity = stationarity;
output.target_val = target_log(k);
output.delta = delta;
output.iterations = k;
output.successful_steps = k - null_count;
output.complex_count = complex_count;
output.alpha = alpha;
output.beta = beta;
output.stat_log = stat_log(1:k+1);
output.target_log = target_log(1:k);

% If specified, print the exit message
if diagnostics
    fprintf('\nThe method terminated with the following output:\n\n');
    disp(output);
end

end