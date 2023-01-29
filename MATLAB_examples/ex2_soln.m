close all; clear; clc

%%%% Example 2: Given a system f(x) = b, solve for x by using grid search

% Load b
load data/ex1_to_3.mat

min_loss = Inf;
for x3 = 0:0.1:1
    for x2 = 0:0.1:1
        for x1 = 0:0.1:1
            x = [x1; x2; x3];
            Fx = applyF(x);
            loss_val = loss(Fx,b);

            if loss_val < min_loss
                min_loss = loss_val;
                best_solution = x;
            end

        end
    end
end

disp('The best x is')
disp(best_solution)


%% Some useful functions

% Compute f(x)
function b = applyF(x)
    f = [1 0 4; 
        0.2 3 1; 
        0 1 0];
    b = f*x;
end

% Compute the difference between two vectors using the Euclidean distance
function loss_val = loss(b_estimated,b)
    loss_val = norm(b_estimated-b,2)^2;
end

