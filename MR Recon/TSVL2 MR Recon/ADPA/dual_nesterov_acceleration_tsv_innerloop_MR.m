function [u, curr_p1, curr_p2] = dual_nesterov_acceleration_tsv(f0, alpha, beta, gamma, iterations)

%close all
%clear
%clc

%Reading the image:
%f0 = imread('smooth1.bmp');
%f0 = f0(:,:,1);
[m,n] = size(f0);
f0 = double(f0);
%figure; imagesc(f0); colormap(gray); axis off; axis equal;

%Setting the parameters
%alpha = 0.1;
%beta = alpha*1e4;
%gamma = 1;
tol = 1e-100;

%ground_truth = load('TSV_alpha01_optimal_soln.mat');
%u_star = ground_truth.TSV_u;
%p_star = ground_truth.TSV_p;
%f_star = Primal_Eval_TSV(u_star, p_star(:,:,1), p_star(:,:,2), f0, alpha, beta, gamma);

%cond_num = (gamma+4*beta)*(8 + (1/4*beta));
%dual_beta = (1-sqrt(1/cond_num))/(1+sqrt(1/cond_num));

%Initialising the variables for the iterations
alpha0 = 1;
qk_1 = zeros(m, n, 2);
qk_2 = zeros(m, n, 2);

%Calculating the diagonal values for the fourier inverse.
[Y,X] = meshgrid(0:n-1,0:m-1);
G1 = cos(2*pi*Y/n)-1;
G2 = cos(2*pi*X/m)-1;
a11 = gamma - 2*beta*(G1);
a22 = gamma - 2*beta*(G2);

tic
stop = [];
for step = 1:iterations
    
    
    %updating p: 
    curr_p1 = ifft2((fft2(alpha*qk_1(:,:,1)))./a11);
    curr_p2 = ifft2((fft2(alpha*qk_1(:,:,2)))./a22);
    
    %Performing proximal gradient updates on the dual variable
    velocity = qk_1 - qk_2;
    alpha1 = (1 + sqrt(1+4*alpha0^2))/2;
    %p = qk_1 + dual_beta * velocity; 
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
        
    %Recording the the energy, and deciding whether to break the loop.
    res = qk_1 - qk_2;
    diff = sum(abs(res(:)))/sum(abs(qk_1(:)));
    stop = cat(2, stop, diff);
    if  diff < tol
        fprintf(' iterate %d times, stop due to converge to tolerance \n', step);
        break
    end
    
    %Calculating the objective function:
    u = f0 + alpha*div(qk_2(:,:,1),qk_2(:,:,2));
    P = Primal_Eval_TSV(u , curr_p1, curr_p2, f0, alpha, beta, gamma);
    D = Dual_Eval_TSV(qk_2 , curr_p1, curr_p2, f0, alpha, beta, gamma);
    Energy(step) = P-D;

end
toc
u = f0 + alpha*div(qk(:,:,1), qk(:,:,2));
%figure; imagesc(u); colormap(gray); axis off; axis equal;
%figure; plot(log(stop));xlabel('Iterations');ylabel('Energy');legend('Energy/Iterations');
%figure; plot(log(real(Energy)))

% Compute proximal gradient
function out = prox_grad(p, d1, d2, f0, alpha, beta)
tau = 1/(8*alpha^2 + alpha^2/(4*beta));
p1 = p(:,:,1);
p2 = p(:,:,2);
div_p = div(p1, p2);
tmp = alpha*div_p + f0;
q1 = p1 + tau*(alpha*(Fx(tmp) - d1));
q2 = p2 + tau*(alpha*(Fy(tmp) - d2));
out = cat(3, q1./max(abs(q1),1), q2./max(abs(q2),1));
 %q = max(sqrt(q1.*conj(q1) + q2.*conj(q2)),1);
 %out = cat(3, q1./q, q2./q);

% Compute divergence using backward derivative
function f = div(a,b)
f = Bx(a) + By(b);

% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
function Fxu = Fx(u)
    Fxu = circshift(u,[0 -1])-u;

%Forward derivative operator on y with boundary condition u(m+1,:)=u(1,:)
function Fyu = Fy(u)
    Fyu = circshift(u,[-1 0])-u;

%Backward derivative operator on x with boundary condition u(:,1)=u(:,n+1)
function Bxu = Bx(u)
    Bxu = u - circshift(u,[0 1]);

%Backward derivative operator on y with boundary condition u(1,:)=u(n+1,:)
function Byu = By(u)
    Byu = u - circshift(u,[1 0]);

function EV = Primal_Eval_TSV(u , p1, p2, f0, alpha, beta, gamma)
    A = 0.5*(u - f0).*conj(u-f0)                                     ...
        + alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2)) ...
        + 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))                ...
        + 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
    
    EV = sum(sum(A));
    
function DEV = Dual_Eval_TSV(q , p1, p2, f0, alpha, beta, gamma)
    div_q = Bx(q(:,:,1)) + By(q(:,:,2));
    A = -0.5*alpha^2*div_q.*conj(div_q)                               ...
    - alpha*(real(f0).*real(div_q) + real(p1).*real(q(:,:,1)) + real(p2).*real(q(:,:,2)))   ...
    -alpha*(imag(f0).*imag(div_q) + imag(p1).*imag(q(:,:,1)) + imag(p2).*imag(q(:,:,2))) ...
    + 0.5*beta*(Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))                  ...
    + 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
        
    DEV = sum(sum(A));
    
    
    