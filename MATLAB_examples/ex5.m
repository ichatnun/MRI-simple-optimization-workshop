% We have seen from the lecture that our system is under-determined. 
% In this example, we include a regularization term to deal with that.

%% Exploration
% 1) Try different learning rate, alpha
% 2) Try different initial solution
% 3) Try different regularization parameter, lambda
% 4) Try different reduction factor, R, and you will see that you need to
% adjust lambda to accommodate the change

close all; clear; clc
set(0,'DefaultFigureWindowStyle','docked')
addpath(strcat(pwd,'/utils_MRI'));

%% Load data
load data/MRI_example/image.mat

im_size = size(x_true);
dynamic_range = [0, 0.5]; % We use this range to display the data in the image space


%% Simulate acquired undersampled k-space data: y = Ax = MFx

% Generate the sampling mask, M, with a specific reduction factor, R
R = 2;
sampling_mask = simulateSamplingMask(R, im_size);

% Simulate the acquired k-space data
y = applyA(x_true, sampling_mask);

% Display some data
figure(1)
subplot(2,2,1), imagesc(abs(x_true)), colormap gray, axis image off, clim(dynamic_range), title('True image')
subplot(2,2,2), imagesc(abs(fft2c(x_true)).^0.1), colormap gray, axis image off, title('Fully-sampled k-space')
subplot(2,2,3), imagesc(sampling_mask), colormap gray, axis image off, title('Sampling mask')
subplot(2,2,4), imagesc(abs(y).^0.1), colormap gray, axis image off, title('Under-sampled k-space')

%% Reconstruct the data
num_iters = 1000; % Number of iterations
x = rand(size(y)); % Initial solution
alpha = 0.05; % Learning rate
loss_curve = zeros(num_iters,1); % Stores the loss values over iterations
lambda = 0.4; % Regularization parameter

% Performing gradient descent
for iter = 1:num_iters

    % Compute the gradient
    grad_data_term = computeGradientDataTerm(x, sampling_mask, y);
    grad_reg_term = computeGradientRegTerm(x);
    grad = grad_data_term + lambda*grad_reg_term;

    % Update x
    x = x - alpha*grad;

    % Compute the loss value
    MFx = applyA(x, sampling_mask);
    loss_data_term = loss(MFx(:),y(:));
    loss_reg_term = norm(x(:),2)^2;
    loss_curve(iter) = loss_data_term + lambda*loss_reg_term;

    % Display intermediate reconstructed images
    if true
        if ((iter < 100) && (mod(iter,10) == 0)) || ((iter>=100) && (mod(iter,100) == 0))
            figure(2)
            subplot(1,2,1)
            imagesc(abs(x)), colormap gray, axis image off, clim(dynamic_range)
            title(['Iter ', num2str(iter)])

            subplot(1,2,2)
            plot(1:iter,loss_curve(1:iter))
            pause(1)
        end
    end

end

% Mask out the irrelevant parts using the ROI mask
x_recon = x.*roi_mask;

%% Display the results
figure(3);
plot(1:num_iters,loss_curve)
xlabel('Iterations'), ylabel('Loss function')

figure(4)
subplot(1,3,1), imagesc(abs(x_true)), colormap gray, axis image off, clim(dynamic_range), title('True image')
subplot(1,3,2), imagesc(abs(x_recon)), colormap gray, axis image off, clim(dynamic_range), title('Reconstructed image')
subplot(1,3,3), imagesc(abs(ifft2c(y))), colormap gray, axis image off, clim(dynamic_range), title('Direct application of IFFT')


%% Useful functions

% Simulate the sampling mask with a specific reduction factor, R
function sampling_mask = simulateSamplingMask(R, im_size)
    pdf = genPDF(im_size, 5, 1/R , 2 , 0.01, 0);
    sampling_mask = genSampling(pdf,10,60);
end

% Compute the Euclidean distance between y_estimated and y
function loss_val = loss(y_estimated,y)
    loss_val = norm(y_estimated(:)-y(:),2)^2;
end

% Compute the gradient of the regularization term: ||x||_2^2
function grad = computeGradientRegTerm(x)
    grad = 2*x;
end

% Compute the gradient of the data term: ||MFx-y||^2_2
function grad = computeGradientDataTerm(x, M, y)

    % Compute Ax-y
    Ax_minus_y = ...

    % grad = 2*A.'*(A*x-y);
    grad = 2 * applyATranspose(Ax_minus_y, M);
end

% Apply the forward model: MFx
function Ax = applyA(x,M)
    Fx = ...
    MFx = M.*Fx;
    Ax = MFx;
end

% Apply A'*input: F'*(M*input) 
function output = applyATranspose(input,M)
    output = ifft2c(conj(M).*input);
end

% Compute the Fourier transform
function Fx = fft2c(x)
    Fx = fftshift(fft2(fftshift(x)));
end

% Compute the inverse Fourier transform
function Fty = ifft2c(y)
    Fty = ifftshift(ifft2(ifftshift(y)));
end

