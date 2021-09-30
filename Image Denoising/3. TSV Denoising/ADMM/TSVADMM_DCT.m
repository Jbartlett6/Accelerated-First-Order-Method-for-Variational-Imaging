function [time, conv] = TSV_ADMM_DCT()
close all
clc
clear

%Loading the Image
source = imread('smooth1.bmp');
%source = imread('lenaNoise.bmp');
f0 = source(:,:,1);
f0 = double(f0)/255;
[m, n] = size(f0);

%Initialisation 
theta = 5  ;
prev_p = zeros(m,n,2);
prev_u = zeros(m,n,1);
prev_w = zeros(m,n,2);
prev_b = zeros(m,n,2);

%Alpha and beta values.
alpha = 0.1;
beta = 1e4*alpha;
gamma = 1;

%Creating the two eigenvalue matricies for the u and p updates:
%u update:
[Y, X] = meshgrid(0:n-1,0:m-1);
G = 2*cos(pi*X/m)+2*cos(pi*Y/n)-4;
G1 = 1 - theta*G; 
[Y,X] = meshgrid(0:n-1,0:m-1);
G11 = cos(pi*Y/n)-1;
G22 = cos(pi*X/m)-1;
eval_x = gamma + theta - 2*beta*G11;
eval_y = gamma + theta - 2*beta*G22;

for i = 1:10000
    %u update:
    curr_u = mirt_idctn(mirt_dctn(f0 - theta*(Bx(prev_w(:,:,1)+prev_p(:,:,1)-prev_b(:,:,1))+By(prev_w(:,:,2)+prev_p(:,:,2)-prev_b(:,:,2))))./G1);
    
    %p update:
    grad_u = cat(3, Fx(curr_u), Fy(curr_u));
    p1 = mirt_idctn(mirt_dctn(-theta*(prev_w(:,:,1) - grad_u(:,:,1) - prev_b(:,:,1)))./eval_x);
    p2 = mirt_idctn(mirt_dctn(-theta*(prev_w(:,:,2) - grad_u(:,:,2) - prev_b(:,:,2)))./eval_y);
    curr_p = cat(3,p1,p2);
    
    %w update:
    curr_w = soft_threshold(grad_u - curr_p+prev_b,alpha/theta);
    
    %b update:
    curr_b = prev_b + grad_u - curr_p - curr_w; 
    
    %Resetting variables for next iteration:
    prev_u = curr_u;
    prev_p = curr_p;
    prev_w = curr_w;
    prev_b = curr_b;
    
   %Recording the function of the objective value: 
   Energy(i) = Primal_Eval_TSV(curr_u, curr_p(:,:,1), curr_p(:,:,2), f0, alpha, beta, gamma);

  
end
figure; imagesc(curr_u); colormap(gray); axis off; axis equal;
figure; plot(log(Energy)); title('Objective Value');


%Defining the forward and backward derivatives for DCT
% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
function Fxu = Fx(u)
[m,n] = size(u);
Fxu = circshift(u,[0 -1])-u;
Fxu(:,n) = zeros(m,1);

% Forward derivative operator on y with boundary condition u(1,:,:)=u(m,:,:)
function Fyu = Fy(u)
[m,n] = size(u);
Fyu = circshift(u,[-1 0])-u;
Fyu(m,:) = zeros(1,n);

% Backward derivative operator on x with boundary condition Bxu(:,1)=u(:,1)
function Bxu = Bx(u)
[~,n] = size(u);
Bxu = u - circshift(u,[0 1]);
Bxu(:,1) = u(:,1);
Bxu(:,n) = -u(:,n-1);

% Backward derivative operator on y with boundary condition Bxu(1,:)=u(1,:)
function Byu = By(u)
[m,~] = size(u);
Byu = u - circshift(u,[1 0]);
Byu(1,:) = u(1,:);
Byu(m,:) = -u(m-1,:);

%Modified Soft Threshold:
%Anisotropic Soft thresholding
function sf = soft_threshold(u,t)
    %Soft thresholding function
    u1 = u(:,:,1);
    u2 = u(:,:,2);
    %Isotropic
    %norm = sqrt(u1.^2+u2.^2+eps);
    %sf = (1./(norm)).*u.*max(norm-t,0);
    %sf = sign(u).*max(norm-t,0);
    
    %Anisotropic
    sf1 = max(abs(u1)-t,0).*sign(u1);
    sf2 = max(abs(u2)-t,0).*sign(u2);
    sf = cat(3,sf1,sf2);
    
%Evaluating the Primal Objective function:    
function EV = Primal_Eval_TSV(u , p1, p2, f0, alpha, beta, gamma)
    A = 0.5*(u - f0).^2                             ...
    + alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2))   ...
    + 0.5*beta * (Fx(p1).^2 + Fy(p2).^2)            ...
    + 0.5*gamma*(p1.^2 + p2.^2);
    EV = sum(sum(A));
    
