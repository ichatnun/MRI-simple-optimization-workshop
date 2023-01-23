close all; clear; clc

%%%% Example 4

% True solution
x_true = [0.5; 0.4; 0.2];

%% System 1
A1 = [1 0 4; 
     0.2 3 1; 
     0 1 0];
b1 = A1*x_true;

x_recon_sys1 = A1\b1;

disp('*************** System 1 ******************')
disp('x_true = ')
disp(x_true)
disp('x_recon from system 1 = ')
disp(x_recon_sys1)
disp('Observed b = ')
disp(b1)
disp('Reconstructed b = ')
disp(A1*x_recon_sys1)


%% System 2
A2 = [1 0 4];
b2 = A2*x_true;

x_recon_sys2 = A2\b2;
disp('*************** System 2 ******************')
disp('x_true = ')
disp(x_true)
disp('x_recon from system 2 = ')
disp(x_recon_sys2)
disp('Observed b = ')
disp(b2)
disp('Reconstructed b = ')
disp(A2*x_recon_sys2)

%% System 2 with different solution
x_recon_sys2_pinv = pinv(A2)*b2;
disp('*************** System 2 with pinv solution ******************')
disp('x_true = ')
disp(x_true)
disp('x_recon from system 2 = ')
disp(x_recon_sys2_pinv)
disp('Observed b = ')
disp(b2)
disp('Reconstructed b = ')
disp(A2*x_recon_sys2_pinv)