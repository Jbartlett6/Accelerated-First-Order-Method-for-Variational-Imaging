function out = Proximal_grad_descent_MR_Recon_TSV()
close all
clear all
clc

%Creating undersampled image - note this is not a completely randomly
%undersampled image.
G_T = imread('test.png');
%G_T = imread('card1.bmp');
%G_T = imread('MR_Toy_example.JPG');
G_T = double(G_T(:,:,1));
[m,n] = size(G_T);
[F,D,U_I] = Undersample(G_T,0.1);


[m,n] = size(F);
scale = sqrt(m*n);

%Initialising the hyperparameters
alpha = 0.075;
beta = alpha*2e2;
gamma = 1;

%Initialising the variables for the loop:
prev_u = zeros(m,n);
prev_u_theta = 1;
u_velocity = zeros(m,n);
for i = 1:100
     %Updating the momentum term for m_hat
    curr_u_theta = (1 + sqrt(1+4*prev_u_theta^2))/2;
    u_momentum = (prev_u_theta - 1)/curr_u_theta;
    u_step = prev_u + u_momentum*u_velocity;
    
    %The outer loop gradient decision 
    curr_u = u_step - scale*ifft2(D.*(D.*fft2(u_step)/scale-F));    
    [im, ~, ~] = dual_nesterov_acceleration_tsv_innerloop_MR(curr_u,alpha, beta, gamma, 100);
    
    %Restart Decision for outer loop
    u_gen_grad = u_step(:)-im(:);
    dot2 = real(u_gen_grad).*real(u_velocity(:)) + imag(u_gen_grad).*imag(u_velocity(:)); 
    
    if sum(dot2,'all') >0; 
        curr_u_theta = 1;
    end
    
    %Recording the energy of the outer loop
    %res = curr_u - prev_u;
    %diff(i) = Primal_Eval_TSV(im , p1, p2, F, D, alpha, beta, gamma);
    
    
    %Resetting the variables for the next iteration
    u_velocity = im-prev_u;
    prev_u_theta = curr_u_theta;
    prev_u = im;
end
out = abs(im);
%figure; imagesc(abs(im)); colormap('gray'); axis off;
%figure; plot(diff - min(diff))
%diff(i)


    function EV = Primal_Eval_TSV(u , p1, p2, F, D, alpha, beta, gamma)
        [m,n] = size(u); 
        scale = sqrt(m*n);
        %A = 0.5*(D.*fft2(u)/scale - F).*conj((D.*fft2(u)/scale - F))    ...
        %+ alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2))                   ...
        %+ 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))      ...
        %+ 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
    
        %Isotropic:
        A = 0.5*(D.*fft2(u)/scale - F).*conj((D.*fft2(u)/scale - F))                        ...
        + alpha * sqrt((Fx(u) - p1).*conj(Fx(u) - p1) + (Fy(u) - p2).*conj(Fy(u) - p2))     ...
        + 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))                          ...
        + 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
        
    EV = sum(sum(A));
   
function [F,D, U_I] = Undersample(image,sparsity)
    [m,n] = size(image);
    
    D = rand(m,n);
    D = double(D<sparsity);
    D(floor((m/4)):floor(3*(m/4)),floor((n/4)):floor(3*(n/4)) ) = 1;
    D = ifftshift(D);
    %Creating the undersampled k-space:
    F = D.*fft2(image)/sqrt(m*n);
    U_I = abs(ifft2(F));
    
    
% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
function Fxu = Fx(u)
    Fxu = circshift(u,[0 -1])-u;

%Forward derivative operator on y with boundary condition u(m+1,:)=u(1,:)
function Fyu = Fy(u)
    Fyu = circshift(u,[-1 0])-u;
