function u = tsvl2_MR_optimizer_easy()
close all
clear all 
clc

%Selecting the hyperparameters
lambda = .3;
theta_1 = 1;   % convergenece parameter
theta_2 = 0.1; % convergenece parameter
beta = 2e2*lambda;
gamma = 1;
[F,D] = create_square_image(125,0.5);

%Initialising the variables at zero:
[m,n]=size(F);
scale = sqrt(m*n);

%Initaising the variables.
b11=zeros(m,n); b12=zeros(m,n);
d1 =zeros(m,n); 
w11=zeros(m,n); w12=zeros(m,n);
p11=zeros(m,n); p12=zeros(m,n);
Wx =zeros(m,n);
u =zeros(m,n);  v =zeros(m,n);

%Creating the matricies for the fft inversion - requires change to apply to
%TSV/TGV.
[Y,X]=meshgrid(0:n-1,0:m-1);
G1=cos(2*pi*Y/n)-1;
G2=cos(2*pi*X/m)-1;
a11=gamma+theta_1-2*beta*(G1);
a22=gamma+theta_1-2*beta*(G2);

for i=1 : 5000
    
    temp_u = u;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u = resolvent_operator(Wx-d1,F, D, theta_2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update u1 (w) using dct
    div_w_b=Bx(w11-b11+p11)+By(w12-b12+p12);
    g=theta_2*(u+d1)-theta_1*div_w_b;
    Wx=ifft2(fft2(g)./(theta_2-2*theta_1*(G1+G2)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update p using fourier inverse:
    p11=ifft2((fft2(theta_1*(Fx(Wx)+b11-w11)))./a11);
    p12=ifft2((fft2(theta_1*(Fy(Wx)+b12-w12)))./a22);

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %Soft Thresholding, to update w:
    c11=Fx(Wx)-p11+b11;
    c12=Fy(Wx)-p12+b12;
   
    %Isotropic
    abs_c=sqrt(c11.*conj(c11)+c12.*conj(c12)+eps);
    w11=max(abs_c-lambda/theta_1,0).*c11./abs_c;
    w12=max(abs_c-lambda/theta_1,0).*c12./abs_c;
   
    %Anisotropic
    %w11=max(abs(c11)-lambda/theta_1,0).*sign(c11);
    %w12=max(abs(c12)-lambda/theta_1,0).*sign(c12);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update Bregman iterative parameters b
    b11=c11-w11;
    b12=c12-w12;
    d1=d1+u-Wx;
   
    %Evaluating the primal objective function
    EV(i) = Primal_Eval_TSV(u , p11, p12, F, D, lambda, beta, gamma);
end
    figure; imagesc(abs(u)); colormap('gray'); title('Reconstructed Image')
    figure; plot(EV);
    EV(i);


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


function u = resolvent_operator(tmp1,F, D, theta_2)
%Updating M using fft inverse method
    [m,n] = size(D);
    scale = sqrt(m*n);
    num = D.*F - theta_2*fft2(-tmp1)/scale;
    u = scale*ifft2(num./(D.*D + theta_2));
    
    %Creating the undersampled image and mask
function [F,D] = create_square_image(N,sparsity)
    % build an image of a square
    image = zeros(N,N);
    image(N/4:3*N/4,N/4:3*N/4)=255;
    
    %Include image here
    %image = imread('MR_Toy_example.JPG');
    %image = imread('smooth_denoised.png');
    image = imread('card1.bmp');
    image = double(image(:,:,1));
    [m,n] = size(image);
    
    % build the sampling matrix, D
    D = rand(m,n);
    D = double(D<sparsity);

    %Creating the undersampled k-space:
    F = D.*fft2(image)/sqrt(m*n);

    %Plotting the undersampled image:
    figure;
    imagesc(abs(ifft2(F)));colormap('gray')
    figure; imagesc(image);colormap('gray'); title('original image');
    
    function EV = Primal_Eval_TSV(u , p1, p2, F, D, alpha, beta, gamma)
        [m,n] = size(u); 
        scale = sqrt(m*n);
        %Anisotropic
        A = 0.5*(D.*fft2(u)/scale - F).*conj((D.*fft2(u)/scale - F))    ...
        + alpha * (abs(Fx(u) - p1) + abs(Fy(u) - p2))                   ...
        + 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))      ...
        + 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
        
        %Isotropic:
        %A = 0.5*(D.*fft2(u)/scale - F).*conj((D.*fft2(u)/scale - F))    ...
        %+ alpha * sqrt((Fx(u) - p1).*conj(Fx(u) - p1) + (Fy(u) - p2).*conj(Fy(u) - p2))               ...
        %+ 0.5*beta * (Fx(p1).*conj(Fx(p1)) + Fy(p2).*conj(Fy(p2)))      ...
        %+ 0.5*gamma*(p1.*conj(p1) + p2.*conj(p2));
        EV = sum(sum(A));