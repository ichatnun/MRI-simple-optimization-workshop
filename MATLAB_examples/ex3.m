close all; clear; clc

%%%% Example 3: Given a system f(x) = b, solve for x by using gradient
%%%% descent

% Load b
load data/ex1_to_3.mat

% Our initial solution
x = [0; 0; 0];

% Apply the gradient descent algorithm
alpha = 0.001; % The learning rate
num_updates = 1000; % Number of updates/iterations
loss_curve = zeros(num_updates,1); % Stores the loss values over iterations

for iter = 1:num_updates
    
    % Compute the gradient using the current solution

    % Update x

    % Compute the current loss value

end

%% Display the loss curve
figure(1)
plot(1:num_updates,loss_curve)

%% Display the final solution
disp('The final solution is')
disp(x)


%% Some useful functions

% Compute f(x)
function b = applyF(x)
    A = [1 0 4; 
        0.2 3 1; 
        0 1 0];
    b = A*x;
end

% Compute the difference between two vectors using the Euclidean distance
function loss_val = loss(b_estimated,b)
    loss_val = norm(b_estimated-b,2)^2;
end

% Compute the gradient of the loss function
function grad = computeGradient(x,b)
    A = [1 0 4; 
        0.2 3 1; 
        0 1 0];
    grad = 
end

