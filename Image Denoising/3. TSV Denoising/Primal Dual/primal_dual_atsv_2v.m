function u = primal_dual_atsv_2v()
%A function to perfom primal dual TSVL2 denoising.

clc
close all
clear

%Reading the images
f0=imread('smooth1.bmp');
%f0 = imread('lenaNoise.bmp');
f0=f0(:,:,1);
[m,n]=size(f0);
f0=double(f0)/255;

%Initialising the Parameters
lambda = .1;
beta = lambda*1e4;
gamma = 1;

%Initialising the step size
sigma = 1/(4*beta+gamma);
tau= 1/(8*lambda^2);
theta = 1;


%Initialising the variables
%x = p
x_k_x = zeros(m,n);
x_k_y = zeros(m,n);

%x_ba = p_ba
xba_k_x = zeros(m,n);
xba_k_y = zeros(m,n);

%y = q
y_k_x = zeros(m,n);
y_k_y = zeros(m,n);

tic
for step = 1:10000
%     step
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %The q step
    diverg= div(y_k_x, y_k_y);
    tmp1 = y_k_x + tau*(lambda*(Fx(lambda*diverg + f0)) - lambda*xba_k_x);
    tmp2 = y_k_y + tau*(lambda*(Fy(lambda*diverg + f0)) - lambda*xba_k_y);
    
    %Anisotropic
    y_k1_x = tmp1./max(abs(tmp1),1);
    y_k2_y = tmp2./max(abs(tmp2),1);

    %Isotropic
    %tmp = max(sqrt(tmp1.^2+tmp2.^2),1);
    %y_k1_x = tmp1./tmp;
    %y_k1_y = tmp2./tmp;
    
    
    %x update
    x_k1_x = x_k_x + sigma*(beta*Bx(Fx(x_k_x)) + lambda*y_k1_x - gamma*x_k_x);
    x_k1_y = x_k_y + sigma*(beta*By(Fy(x_k_y)) + lambda*y_k2_y - gamma*x_k_y);
    
    %x_ba update
    xba_k1_x = x_k1_x + theta*(x_k1_x - x_k_x);
    xba_k1_y = x_k1_y + theta*(x_k1_y - x_k_y);
    
    %y update
    y = cat(3,y_k1_x,y_k2_y);
    
   
    %Updating the variables for the next iteration%
    x_k_x   = x_k1_x;
    x_k_y   = x_k1_y;
    xba_k_x = xba_k1_x;
    xba_k_y = xba_k1_y;
    y_k_x = y_k1_x;
    y_k_y = y_k2_y;
    
    %Calculating the value of the objective function
    u = f0 + lambda*div(y_k_x, y_k_y);
    Energy_primal(step) = Primal_Eval_TSV(u , x_k_x, x_k_y, f0, lambda, beta, gamma);
end
u = f0 + lambda*div(y_k_x,y_k_y);
figure; imagesc(f0); colormap(gray); axis off; axis equal;
figure; imagesc(u); colormap(gray); axis off; axis equal;
figure; plot(log(Energy_primal(1:end)));xlabel('Iterations');ylabel('Energy');legend('Energy/Iterations');
end

% Compute divergence using backward derivative
function f = div(a,b)
f = Bx(a)+By(b);
end

% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
function Fxu = Fx(u)
[m,n] = size(u);
Fxu = circshift(u,[0 -1])-u;
Fxu(:,n) = zeros(m,1);
end
% Forward derivative operator on y with boundary condition u(1,:,:)=u(m,:,:)
function Fyu = Fy(u)
[m,n] = size(u);
Fyu = circshift(u,[-1 0])-u;
Fyu(m,:) = zeros(1,n);
end

% Backward derivative operator on x with boundary condition Bxu(:,1)=u(:,1)
function Bxu = Bx(u)
[~,n] = size(u);
Bxu = u - circshift(u,[0 1]);
Bxu(:,1) = u(:,1);
Bxu(:,n) = -u(:,n-1);
end

% Backward derivative operator on y with boundary condition Bxu(1,:)=u(1,:)
function Byu = By(u)
[m,~] = size(u);
Byu = u - circshift(u,[1 0]);
Byu(1,:) = u(1,:);
Byu(m,:) = -u(m-1,:);
end

%A function to evaluate the primal TSV denoising objective
function EV = Primal_Eval_TSV(u , p1, p2, f0, alpha, beta, gamma)
    A = 0.5*(u - f0).^2                                     ...
    + alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2)) ...
    + 0.5*beta * (Fx(p1).^2 + Fy(p2).^2)                ...
    + 0.5*gamma*(p1.^2 + p2.^2);
    
    EV = sum(sum(A));
end



    