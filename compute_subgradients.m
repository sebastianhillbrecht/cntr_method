function [subgrads, num_subgrads, it] = compute_subgradients(OmegaPlusMat, ...
    OmegaZeroMat, grad_pi, grad_mu1, epsilon, gamma, max_number)
%COMPUTE_SUBGRADIENTS WARNING: This method is highly inefficient and
%likely to fail in dimensions n1,n2 > 50!
%
%Computes a given number of elements of the Bouligand subdifferential of a
%given point by iterating over every possible contruction of subsets of the
%biactive set
%
% Inputs:
%   - OmegaPlusMat: Matrix corresponding to the active set associated with the given point (matrix)
%   - OmegaZeroMat: Matrix corresponding to the biactive set associated with the given point (matrix)
%   - grad_pi:      Target function's gradient w.r.t. the state pi at the
%                   given point (matrix)
%   - grad_mu1:     Target function's gradient w.r.t. the control mu1 at
%                   the given point (vector)
%   - epsilon:      Kantorovich regularization parameter (positive scalar)
%   - gamma:        Dual problem regularization parameter (positive scalar)
%   - max_number:   Maximum number of subgradients that shall be computed (positive integer)
%
% Outputs:
%   - subgrads:     List of computed subgradients (matrix)
%   - num_subgrads: Number of subgrads computed (positive integer)
%   - it:           Number of iterations needed (positive integer)

% Fetch dimensions
[n1, n2] = size(OmegaPlusMat);

% Choose the subgradient corresponding to A = {} to be the first element of
% the list of subgradients
M = OmegaPlusMat .* grad_pi;
sysMat = [diag(OmegaPlusMat * ones(n2,1)), OmegaPlusMat; ...
          OmegaPlusMat',                   diag(OmegaPlusMat' * ones(n1,1))] ...
            + gamma * epsilon * eye(n1+n2);
rhs = [sum(M,2); sum(M,1)'];
gh = sysMat \ rhs;
subgrads(:,1) = gh(1:n1);

% Fetch the row and column indices of the elements of the biactive set
% Omega0
i_ind = nonzeros(any(OmegaZeroMat') .* (1:n1));
j_ind = nonzeros(any(OmegaZeroMat) .* (1:n2));

% Initialization
it = 0;
flag1 = 0;
flag2 = 0;
flag3 = 0;
tmp = 2;
for i = 1:numel(i_ind)
    % Compute every possible subset of i_ind that has length i
    combs_i = nchoosek(i_ind, i);

    % Iterate over constructions of v1
    for l_i = 1:size(combs_i, 1)
        v1 = zeros(n1, 1);
        v1(combs_i(l_i, :)) = 1;
        v1(setdiff(i_ind, combs_i(l_i, :))) = -1;
        
        % Combine each construction v1 with a seperate construction v2
        for j = 0:numel(j_ind)
            % Compute every possible subset of j_ind that has length j
            combs_j = nchoosek(j_ind, j);
            
            % Iterate over constructions of v2
            for l_j = 1:size(combs_j, 1)
                v2 = zeros(n2,1);
                v2(combs_j(l_j, :)) = -1.5;

                % Compute Cartesian sum to determine the resulting set A
                V = v1 + v2';
                if any(V(OmegaZeroMat) == 0, "all")
                    % Skip this set, as it does not satisfy the
                    % requirements!
                else
                    % Fetch the matrix corresponding to the set A
                    AMat(:, :, it+1) = (V .* OmegaZeroMat) > 0;
                    ASet(it+1, AMat(:, :, it+1)) = 1;
                    
                    % If the current set A was not considered before,
                    % calculate the corresponding subgradient
                    if it == 0 || ~any(ismember(ASet(1:it, :), ASet(it+1, :), 'rows'))
                        % Construct the matrix corresponding to the set
                        % Omega0 \cup A
                        OmegaPlusAMat = OmegaPlusMat + AMat(:, :, it+1);
                        M = OmegaPlusAMat .* grad_pi;

                        % Calculate the corresponding subgradient
                        sysMat = [diag(OmegaPlusAMat * ones(n2,1)), OmegaPlusAMat; ...
                            OmegaPlusAMat', diag(OmegaPlusAMat' * ones(n1,1))] + gamma * epsilon * eye(n1+n2);
                        rhs = [sum(M,2); sum(M,1)'];
                        gh = sysMat \ rhs;
                        subgrads(:,tmp) = gh(1:n1) + grad_mu1;

                        % Remove subgradients that are identical or too close to each other
                        unique_TOL = 1e-5; % This tolerance depends on the problem structure
                        subgrads = uniquetol(subgrads', unique_TOL, 'ByRows', true)';

                        % Termination
                        if (size(subgrads,2) > max_number) || (it > 100)
                            flag1 = 1;
                            break;
                        end

                        tmp = size(subgrads,2) + 1;
                    end
                    if it > 100
                        flag1 = 1;
                        break;
                    end

                    it = it + 1;
                end
            end
            if flag1
                flag2 = 1;
                break;
            end
        end
        if flag2
            flag3 = 1;
            break;
        end
    end
    if flag3
        break;
    end
end
num_subgrads = size(subgrads,2);

% % Throw a warning, if too few subgradients could be found.
% if size(subgrads,2) < max_number
%     warning(['Unable to find desired amount of subgradients in ', ...
%         num2str(it), ' iterations! Only found ', num2str(size(subgrads,2)), ...
%         ' of ', num2str(max_number), '.'])
% end