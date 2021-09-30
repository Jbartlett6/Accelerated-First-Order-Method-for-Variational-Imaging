function [u,v, curr_p11, curr_p12, curr_p21, curr_p22] = dual_nesterov_acceleration_tsv(f0, f1, alpha, beta, gamma, iterations)

%close all
%clear
%clc

%Reading the image:
%f0 = imread('smooth1.bmp');
%f0 = f0(:,:,1);
[m,n] = size(f0);
%f0 = double(f0)/255;
%figure; imagesc(f0); colormap(gray); axis off; axis equal;

%Setting the parameters
%alpha = 0.1;
%beta = alpha*1e4;
%gamma = 1;
tol = 1e-30;

%ground_truth = load('TSV_alpha01_optimal_soln.mat');
%u_star = ground_truth.TSV_u;
%p_star = ground_truth.TSV_p;
%f_star = Primal_Eval_TSV(u_star, p_star(:,:,1), p_star(:,:,2), f0, alpha, beta, gamma);

%cond_num = (gamma+4*beta)*(8 + (1/4*beta));
%dual_beta = (1-sqrt(1/cond_num))/(1+sqrt(1/cond_num));

%Initialising the variables for the iterations
alpha0 = 1;
qk1_1 = zeros(m, n, 2);
qk1_2 = zeros(m, n, 2);

qk2_1 = zeros(m, n, 2);
qk2_2 = zeros(m, n, 2);

%Calculating the diagonal values for the fourier inverse.
[Y,X] = meshgrid(0:n-1,0:m-1);
G1 = cos(pi*Y/n)-1;
G2 = cos(pi*X/m)-1;
a11 = gamma - 2*beta*(G1);
a22 = gamma - 2*beta*(G2);

tic
stop = [];
for step = 1:iterations
    
    
    %updating p: 
    curr_p11 = mirt_idctn((mirt_dctn(alpha*qk1_1(:,:,1)))./a11);
    curr_p12 = mirt_idctn((mirt_dctn(alpha*qk1_1(:,:,2)))./a22);
    
    curr_p21 = mirt_idctn((mirt_dctn(alpha*qk2_1(:,:,1)))./a11);
    curr_p22 = mirt_idctn((mirt_dctn(alpha*qk2_1(:,:,2)))./a22);
    %Performing proximal gradient updates on the dual variable
    velocity_1 = qk1_1 - qk1_2;
    velocity_2 = qk2_1 - qk2_2;
    
    alpha1 = (1 + sqrt(1+4*alpha0^2))/2;
    %p = qk_1 + dual_beta * velocity; 
    %p1 = qk1_1 + (alpha0-1)/alpha1 * velocity_1;
    %p2 = qk2_1 + (alpha0-1)/alpha1 * velocity_2;
    p1 = qk1_1 + (1-1)/alpha1 * velocity_1;
    p2 = qk2_1 + (1-1)/alpha1 * velocity_2;
    [qk1,qk2] = prox_grad(p1, curr_p11 , curr_p12, f0, p2, curr_p21, curr_p22, f1, alpha, beta);
    
    %restart decision for the dual variable:
    if sum((p1(:)-qk1(:)).*velocity_1(:))+sum((p2(:)-qk2(:)).*velocity_2(:))>0
        alpha1 = 1;
    end

    %Updating the variables for the next iteration.
    qk1_2 = qk1_1;
    qk1_1 = qk1;
    
    qk2_2 = qk2_1;
    qk2_1 = qk2;
    alpha0 = alpha1;
        
    %Recording the the energy, and deciding whether to break the loop.
    res1 = qk1_1 - qk1_2;
    res2 = qk2_1 - qk2_2;
    diff = sum(abs(res1(:)))/sum(abs(qk1_1(:))) + sum(abs(res2(:)))/sum(abs(qk2_1(:)));
    stop = cat(2, stop, diff);
    if  diff < tol
        fprintf(' iterate %d times, stop due to converge to tolerance \n', step);
        break
    end
    
    %Calculating the objective function:
    %u = f0 + alpha*div(qk1_2(:,:,1),qk1_2(:,:,2));
    %P = Primal_Eval_TSV(u , curr_p11, curr_p12, f0, alpha, beta, gamma);
    %D = Dual_Eval_TSV(qk1_2 , curr_p11, curr_p12, f0, alpha, beta, gamma);
    %Energy(step) = P-D;
    
    %u = f1 + alpha*div(qk2_2(:,:,1),qk2_2(:,:,2));
    %P = Primal_Eval_TSV(u , curr_p21, curr_p22, f1, alpha, beta, gamma);
    %D = Dual_Eval_TSV(qk2_2 , curr_p21, curr_p22, f1, alpha, beta, gamma);
    %Energy(step) = P-D;
end
toc
u = f0 + alpha*div(qk1(:,:,1), qk1(:,:,2));
v = f1 + alpha*div(qk2(:,:,1), qk2(:,:,2));
%figure; imagesc(v); colormap(gray); axis off; axis equal;
%figure; plot(log(stop));xlabel('Iterations');ylabel('Energy');legend('Energy/Iterations');
%figure; plot(log(Energy))

% Compute proximal gradient
function [out1, out2] = prox_grad(p1, d11, d12, f0, p2, d21, d22, f1, alpha, beta)
tau = 1/(8*alpha^2 + alpha^2/(4*beta));
p11 = p1(:,:,1);
p12 = p1(:,:,2);
p21 = p2(:,:,1);
p22 = p2(:,:,2);

div_p1 = div(p11, p12);
div_p2 = div(p21, p22);

tmp1 = alpha*div_p1 + f0;
tmp2 = alpha*div_p2 + f1;

q11 = p11 + tau*(alpha*(Fx(tmp1) - d11));
q12 = p12 + tau*(alpha*(Fy(tmp1) - d12));
q21 = p21 + tau*(alpha*(Fx(tmp2) - d21));
q22 = p22 + tau*(alpha*(Fy(tmp2) - d22));
%out = cat(3, q1./max(abs(q1),1), q2./max(abs(q2),1));
 q = max(sqrt(q11.^2 + q12.^2 + q21.^2 + q22.^2),1);
 out1 = cat(3, q11./q, q12./q);
 out2 = cat(3, q21./q, q22./q);
 %out1 = cat(3,q11./max(abs(q11),1),q12./max(abs(q12),1));
 %out2 = cat(3,q21./max(abs(q21),1),q22./max(abs(q22),1));

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

function EV = Primal_Eval_TSV(u , p1, p2, f0, alpha, beta, gamma)
    A = 0.5*(u - f0).^2                                     ...
        + alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2)) ...
        + 0.5*beta * (Fx(p1).^2 + Fy(p2).^2)                ...
        + 0.5*gamma*(p1.^2 + p2.^2);
    
    EV = sum(sum(A));
    
function DEV = Dual_Eval_TSV(q , p1, p2, f0, alpha, beta, gamma)
    div_q = Bx(q(:,:,1)) + By(q(:,:,2));
    A = -0.5*alpha^2*div_q.^2                               ...
    - alpha*(f0.*div_q + p1.*q(:,:,1) + p2.*q(:,:,2))   ...
    + 0.5*beta*(Fx(p1).^2 + Fy(p2).^2)                  ...
    + 0.5*gamma*(p1.^2 + p2.^2);
        
    DEV = sum(sum(A));
    
    
    