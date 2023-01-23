close all; clear; clc

%%%% Example 1: Given a system f(x) = b, solve for x by "guessing"

% Load b
load data/ex1_to_3.mat

% Input your guess and evaluate how good your guess is
x = [-10; 20; 10];


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

