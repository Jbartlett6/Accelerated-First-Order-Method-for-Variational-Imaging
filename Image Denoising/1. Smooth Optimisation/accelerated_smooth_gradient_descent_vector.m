function scores = smooth_gradient_descent_vector()
close all
clear 

%Reading and plotting the image:
source = double(imread('example.png'));
f0 = source(:,:,1) / 255;
[m, n] = size(f0);
figure(1); imagesc(f0); colormap(gray); title('original source image'); axis off; axis equal;

%Reshape image to vector which is column-major
f0 = f0(:); 
u = f0;

%Producing the finite difference matricies:
Fx = kron(fwdiff1D(n),speye(m)); % kronecker product gives 2D x forward diff
Fy = kron(speye(n),fwdiff1D(m)); % kronecker product gives 2D y forward diff
iden = speye(m*n, m*n); % sparse identity

%Initialising the smoothing parameter:
lambda = 10; 
scores = Eval(u,lambda,f0,Fx,Fy);

tmp = iden + lambda*(Fx'*Fx + Fy'*Fy); % tranpose is equivplent to 2D backward diff
% Fx'*Fx + Fy'*Fy is a highly structured matrix, whose max eign is 8 and min eig 0

%Calculating the solution and evlauating the objective function at this
%value:
soln = tmp\f0;
f_star = Eval(soln,lambda,f0,Fx,Fy);

%Setting the condition number for this example:
cond_num = lambda*8 + 1;

%Initialising the variables for the algorithm
prev_theta = 1;
prev_x = zeros(65536,1);
prev_y = prev_x;

%set q = 1 for gradient descent:
%q = 0;
q = 1;
beta = (1-sqrt(1/cond_num))/(1+sqrt(1/cond_num));

% step size determined by max eigenvalue which is equal to lambda*8+1
step_size = 1/(lambda*8+1); 

for i = 1 : 100
    %Performing the greadient update:
    grad_u = (iden + lambda*(Fx'*Fx + Fy'*Fy))*prev_y - f0; 
    curr_x = prev_y-step_size*grad_u;
    
    %Updating the momentum parameter:
    curr_theta = (-(prev_theta^2-q)+sqrt((prev_theta^2-q)^2+4*prev_theta^2))/2;
    curr_beta = prev_theta*(1-prev_theta)/(prev_theta^2+curr_theta);
    
    %Performing the step forward step:
    curr_y = curr_x + curr_beta*(curr_x-prev_x);
    
    %Evaluating the score:
    scores(i) = Eval(curr_x,lambda,f0,Fx,Fy) - f_star;
    %beta_score(i) = curr_beta;
    
    %Evaluating the restart decision 
    grad_u = (iden + lambda*(Fx'*Fx + Fy'*Fy))*curr_y - f0; 
    if sum(grad_u(:).*(curr_x(:)-prev_x(:)))>0
        curr_theta = 1;
        curr_y = curr_x;
    end
    
    %Resetting the variables for the next iteration
    prev_theta = curr_theta;
    prev_x = curr_x;
    prev_y = curr_y;
   
    %Continuously plotting the current iterate:
    img_2D = reshape(curr_x, m, n);
    figure(2); imagesc(img_2D); colormap(gray); title('iterative process'); axis off; axis equal;
    pause(0.001)
    
end
%Plotting the final iterate and plots:
figure(3); imagesc(img_2D); colormap(gray); title('final smoothed image'); axis off; axis equal;
figure(4); plot(log(scores))

%Function used to create the finite difference matrix.
function L = fwdiff1D(N)

b1 = zeros(1,N)-1;
b1(1,N) = 0;

L0 = sparse(1:N, 1:N, b1, N, N);

clear b1
b1 = ones(1, N-1);
L1 = sparse(1:N-1, 2:N, b1, N, N);

L = L0 + L1;

%Evaluating the objective function
function Ev = Eval(u,lambda,f0,Fx,Fy)
dxu = Fx*u;
dyu = Fy*u;
gradu = sum(dxu.^2+dyu.^2);
Ev = 0.5*norm(u-f0).^2+0.5*lambda*gradu;  




