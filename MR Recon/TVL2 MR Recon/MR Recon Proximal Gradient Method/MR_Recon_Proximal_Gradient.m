function MR_Recon_Proximal_Gradient()
close all
clear

%Creating the undersampled square image
[F,D] = create_square_image(128,0.25);
[m,n] = size(ifft2(F));

%Initalising the hyperparameters:
lambda = 10;
p_step = 1/(8*lambda^2);

%Initialising the scale factor for the fft.
scale = sqrt(m*n);

%Initialising m for the outer loop:
prev_m = zeros(m,n);
prev_m_theta = 1;
m_velocity = zeros(m,n);

tic
for i = 1:50
    %Updating the momentum for the outer loop:
    curr_m_theta = (1 + sqrt(1+4*prev_m_theta^2))/2;
    m_momentum = (prev_m_theta - 1)/curr_m_theta;
    m_step = prev_m + m_momentum*m_velocity;
    
    %Gradient update for the outer loop:
    curr_m_hat = m_step - ifft2(D.*D.*fft2(m_step)) + ifft2(D.*F)*scale;
    
    %Initialising p for the inner loop:
    prev_p = zeros(m,n,2);
    alpha0 = 1;
    p_velocity = zeros(m,n,2);
    for j = 1:20
        %p update:
        %Updating the momentum for the inner loop:
        alpha1 = (1 + sqrt(1+4*alpha0^2))/2;
        p_momentum = (alpha0 - 1)/alpha1;
        step_p = prev_p + p_momentum*p_velocity;
        
        %Performing proximal gradient update for the inner loop:
        p_grad = lambda^2*grad(div(step_p)) + lambda*grad(curr_m_hat);
        p_hat = step_p + p_step*p_grad;
        curr_p = proj(p_hat);
        
        %Calculating the energy of the inner loop
        res = curr_p - prev_p;
        diff(j) = sum(abs(res(:)))/sum(abs(curr_p(:)));
        
        %Restart decision for the inner loop:
        gen_grad = (step_p(:)-curr_p(:));
        dot1 = real(gen_grad).*real(p_velocity(:)) + imag(gen_grad).*imag(p_velocity(:));
        if sum(dot1,'all') >0 
            alpha1 = 1;
        end
        
        %Resetting p for the next iteration
        alpha0 = alpha1;
        p_velocity = curr_p - prev_p;
        prev_p = curr_p;
    end
   
    %Updating m using the inner loop proximal step
    curr_m = curr_m_hat + lambda*div(curr_p);
    res = curr_m - prev_m;
    diff_1(i) = sum(abs(res(:)))/sum(abs(curr_m(:)));
    
    %Restart Decision for outer loop
    m_gen_grad = m_step(:)-curr_m(:);
    dot2 = real(m_gen_grad).*real(m_velocity(:)) + imag(m_gen_grad).*imag(m_velocity(:)); 
    if sum(dot2,'all') >0; 
        curr_m_theta = 1;
    end
    
    %Resetting the m variables for the next iteration:
    m_velocity = curr_m - prev_m;
    prev_m = curr_m;
    prev_m_theta = curr_m_theta;
end
toc
figure; imagesc(abs(curr_m)); colormap('gray');
figure; plot(log(diff_1)); title('Outer Loop');
figure; plot(log(diff)); title('Inner Loop');

%Divergence function
function d = div(a)
    d = Bx(a(:,:,1)) + By(a(:,:,2));

%Gradient function
function g = grad(a)
    g = cat(3,Fx(a),Fy(a));

    function p = proj(input)
a = input(:,:,1);
b = input(:,:,2);
%Anisotropic
p1 = a./max(abs(a),1);
p2 = b./max(abs(b),1);
p = cat(3,p1,p2);

%Isotropic
%norm = sqrt((a.*conj(a) + b.*conj(b)));
%de = max(norm,1);
%p = cat(3,a./de,b./de);
%Defining the forward and backward derivatives with periodic boundary
%conditions
%Forward derivative operator on x with boundary condition u(:,n+1)=u(:,1)
function Fxu = Fx(u)
    [m,n] = size(u);
    Fxu = circshift(u,[0 -1])-u;

%Forward derivative operator on y with boundary condition u(m+1,:)=u(1,:)
function Fyu = Fy(u)
    [m,n] = size(u);
    Fyu = circshift(u,[-1 0])-u;

%Backward derivative operator on x with boundary condition u(:,1)=u(:,n+1)
function Bxu = Bx(u)
    [~,n] = size(u);
    Bxu = u - circshift(u,[0 1]);

%Backward derivative operator on y with boundary condition u(1,:)=u(n+1,:)
function Byu = By(u)
    [m,~] = size(u);
    Byu = u - circshift(u,[1 0]);
    
    %Creating the undersampled image and mask
function [F,D] = create_square_image(N,sparsity)
    % build an image of a square
    image = zeros(N,N);
    image(N/4:3*N/4,N/4:3*N/4)=255;
    
    %image = imread('card1.bmp');
    image = imread('MR_Toy_example.JPG');
    %image = imread('smooth_denoised.png');
    image = double(image(:,:,1));
    
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