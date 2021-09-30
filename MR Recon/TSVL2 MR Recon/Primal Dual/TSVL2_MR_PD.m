function TSVL2_MR_PD()
close all 
clear 
clc

%Create undersampled image
[F,D] = create_square_image(125,0.25);

%Initialising hyper-parameters
lambda = .1;
beta = 1e4*lambda;
gamma = 1;

%Initialising step sizes:
u_step_size = 1/(8*lambda^2);
p_step_size = (1/(4*beta + gamma));
q_step_size = 0.01;

%Initialising the variables at zero:
[m,n]=size(F);
scale = sqrt(m*n);

prev_u = zeros(m,n);
prev_p1 = zeros(m,n); prev_p2 = zeros(m,n);
prev_q1 = zeros(m,n); prev_q2 = zeros(m,n);

for i = 1:10000
    %u proximal gradient update:
    u_tilde = prev_u + u_step_size*lambda*div(prev_q1,prev_q2);
    curr_u = prox_u(u_tilde, F, D, 1/u_step_size);
    
    %p gradient update:
    curr_p1 = prev_p1 - p_step_size*(-lambda*prev_q1 - beta*Bx(Fx(prev_p1))+gamma*prev_p1);
    curr_p2 = prev_p2 - p_step_size*(-lambda*prev_q2 - beta*By(Fy(prev_p2))+gamma*prev_p2);
    
    %Interpolation step
    u_ba = curr_u; %+ (curr_u - prev_u);
    p1_ba = curr_p1; %+ (curr_p1 - prev_p1);
    p2_ba = curr_p2; %+ (curr_p2 - prev_p2);
    
    %q gradient update:
    q1_tilde = prev_q1 + q_step_size*(Fx(u_ba) - p1_ba);
    q2_tilde = prev_q2 + q_step_size*(Fy(u_ba) - p2_ba);
    
    %q proximal update:
    [curr_q1, curr_q2] = proj(q1_tilde, q2_tilde);
    
    %Calculating the primal objective function
    energy(i) = Primal_Eval_TSV(curr_u , curr_p1, curr_p2, F, D, lambda, beta, gamma);
    
    %Resetting the variables for the next iterate.
    prev_u = curr_u;
    prev_p1 = curr_p1; prev_p2 = curr_p2;
    prev_q1 = curr_q1; prev_q2 = curr_q2;
  
end
figure; imagesc(abs(curr_u)); colormap('gray');
figure; plot(energy)
energy(i)

%Defining the forward and backward derivatives with periodic boundary
%conditions
%Forward derivative operator on x with boundary condition u(:,n+1)=u(:,1)
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

%Divergence Function
function d = div(a,b)
    d = Bx(a) + By(b);


function [F,D] = create_square_image(N,sparsity)
    % build an image of a square
    %image = zeros(N,N);
    %image(N/4:3*N/4,N/4:3*N/4)=255;
    image = imread('MR_Toy_example.JPG');
    %image = imread('card1.bmp');
    %image = imread('smooth_denoised.png');
    image = double(image(:,:,1));
    image = imresize(image,0.5);
    [m,n] = size(image);
    % build the sampling matrix, D
    
    rng('default')
    D = rand(m,n);
    D = double(D<sparsity);

    %Creating the undersampled k-space:
    F = D.*fft2(image)/sqrt(m*n);

    %Plotting the undersampled image:
    figure;
    imagesc(abs(ifft2(F)));colormap('gray')
    
    
    function u = prox_u(tmp1,F, D, theta_2)
        %Updating M using fft inverse method
        [m,n] = size(D);
        scale = sqrt(m*n);
        num = D.*F + theta_2*fft2(tmp1)/scale;
        u = scale*ifft2(num./(D.*D + theta_2));
        
function [q1,q2] = proj(a,b)
%Anisotropic
%q1 = a./max(abs(a),1);
%q2 = b./max(abs(b),1);
%p = cat(3,p1,p2);

%Isotropic
norm = sqrt((a.*conj(a) + b.*conj(b)));
de = max(norm,1);
q1 = a./de;
q2 = b./de;

    function EV = Primal_Eval_TSV(u , p1, p2, F, D, alpha, beta, gamma)
        [m,n] = size(u); 
        scale = sqrt(m*n);
        %A = 0.5*(D.*fft2(u)/scale - F).*conj((D.*fft2(u)/scale - F))    ...
        %+ alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2))                   ...
        %+ 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))      ...
        %+ 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
    
        %Isotropic:
        A = 0.5*(D.*fft2(u)/scale - F).*conj((D.*fft2(u)/scale - F))    ...
        + alpha * sqrt((Fx(u) - p1).*conj(Fx(u) - p1) + (Fy(u) - p2).*conj(Fy(u) - p2))               ...
        + 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))      ...
        + 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
        
    EV = sum(sum(A));
    