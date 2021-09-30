function u = dual_nesterov_acceleration_tsv()
close all
clear
clc

%Reading the image:
f0 = imread('smooth1.bmp');
%f0 = imread('lenaNoise.bmp')
f0 = f0(:,:,1);
[m,n] = size(f0);
f0 = double(f0)/255;
figure; imagesc(f0); colormap(gray); axis off; axis equal;

%Setting the parameters
alpha = 0.1;
beta = alpha*1e4;
gamma = 1;

%Initialising the variables for the iterations
alpha0 = 1;
qk_1 = zeros(m, n, 2);
qk_2 = zeros(m, n, 2);

%Calculating the diagonal values for the fourier inverse.
[Y,X] = meshgrid(0:n-1,0:m-1);
G1 = cos(pi*Y/n)-1;
G2 = cos(pi*X/m)-1;
a11 = gamma - 2*beta*(G1);
a22 = gamma - 2*beta*(G2);


for i = 1:10000
    %updating p: 
    curr_p1 = mirt_idctn((mirt_dctn(alpha*qk_1(:,:,1)))./a11);
    curr_p2 = mirt_idctn((mirt_dctn(alpha*qk_1(:,:,2)))./a22);
    
    %Performing proximal gradient updates on the dual variable
    velocity = qk_1 - qk_2;
    alpha1 = (1 + sqrt(1+4*alpha0^2))/2;
    p = qk_1 + (alpha0-1)/alpha1 * velocity;
    qk = prox_grad(p, curr_p1 , curr_p2, f0, alpha, beta);
   
    %restart decision for the dual variable:
    if sum((p(:)-qk(:)).*velocity(:))>0
        alpha1 = 1;
    end

    %Updating the variables for the next iteration.
    qk_2 = qk_1;
    qk_1 = qk;
    alpha0 = alpha1;
    
    %Calculating the duality gap:
    u = f0 + alpha*div(qk_2(:,:,1),qk_2(:,:,2));
    P = Primal_Eval_TSV(u , curr_p1, curr_p2, f0, alpha, beta, gamma);
    D = Dual_Eval_TSV(qk_2 , curr_p1, curr_p2, f0, alpha, beta, gamma);
    Energy(i) = P - D;

    
    
end
u = f0 + alpha*div(qk(:,:,1), qk(:,:,2));
figure; imagesc(u); colormap(gray); axis off; axis equal;
figure; plot(log(Energy))

% Compute proximal gradient
function out = prox_grad(p, d1, d2, f0, alpha, beta)
tau = 1/(8*alpha^2 + alpha^2/(4*beta));
p1 = p(:,:,1);
p2 = p(:,:,2);
div_p = div(p1, p2);
tmp = alpha*div_p + f0;
q1 = p1 + tau*(alpha*(Fx(tmp) - d1));
q2 = p2 + tau*(alpha*(Fy(tmp) - d2));
%Anisotropic
out = cat(3, q1./max(abs(q1),1), q2./max(abs(q2),1));
%Isotropic
% q = max(sqrt(q1.^2 + q2.^2),1);
% out = cat(3, q1./q, q2./q);

% Compute divergence using backward derivative
function f = div(a,b)
f = Bx(a) + By(b);

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

%Evaluating the TSV denoising Primal function
function EV = Primal_Eval_TSV(u , p1, p2, f0, alpha, beta, gamma)
    A = 0.5*(u - f0).^2                                     ...
        + alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2))       ...
        + 0.5*beta * (Fx(p1).^2 + Fy(p2).^2)                ...
        + 0.5*gamma*(p1.^2 + p2.^2);
    
    EV = sum(sum(A));
    
%Evaluating the TSV denoising Dual function 
function DEV = Dual_Eval_TSV(q , p1, p2, f0, alpha, beta, gamma)
    div_q = Bx(q(:,:,1)) + By(q(:,:,2));
    A = -0.5*alpha^2*div_q.^2                           ...
    - alpha*(f0.*div_q + p1.*q(:,:,1) + p2.*q(:,:,2))   ...
    + 0.5*beta*(Fx(p1).^2 + Fy(p2).^2)                  ...
    + 0.5*gamma*(p1.^2 + p2.^2);
        
    DEV = sum(sum(A));
    
    
    