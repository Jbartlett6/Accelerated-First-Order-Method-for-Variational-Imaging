function MR_Recon_Chambolle()
close all
clear
%Initialising the test image and undersampling mask:
[F,D] = create_square_image(128,0.25);
[m,n] = size(ifft2(F));
scale = sqrt(m*n);

%Initialising hyperparameters
lambda = 0.1*sqrt(m*n);
theta = 1;
m_step = 1/(2*lambda);
p_step = 1/sqrt(8);
 
%Initialising Variables 
prev_p = zeros(m,n,2);
prev_m = zeros(m,n);
m_bar = zeros(m,n);

%Update loop
tic
for i = 1:1000
    %P step
    p_tilde = prev_p + p_step*grad(m_bar);
    curr_p = proj(p_tilde);
    
    %M step
    m_grad = lambda*ifft2(D.*D.*(fft2(prev_m))-F) - div(curr_p);
    curr_m = prev_m - m_step*m_grad;
    
    %Interpolation step
    m_bar = curr_m + theta*(curr_m - prev_m);
    
    %Calculating energy
    res = curr_m - prev_m;
    diff_1(i) = sum(abs(res(:)))/sum(abs(curr_m(:)));
    
    %Updating Variables for the next iteration
    prev_m = curr_m;
    prev_p = curr_p;
end
toc
figure; imagesc(abs(curr_m)); colormap('gray');
figure; plot(log(diff_1));

function p = proj(input)
    x = input(:,:,1);
    y = input(:,:,2);
    
    %Anisotropic
    %p1 = a./max(abs(a),1);
    %p2 = b./max(abs(b),1);
    %p = cat(3,p1,p2);
    
    %Isotropic
    norm = sqrt((x.*conj(x) + y.*conj(y)));
    de = max(norm,1);
    p = cat(3,x./de,y./de);

%Gradient Function
function g = grad(a)
    g = cat(3,Fx(a),Fy(a));

%Divergence Function 
function d = div(a)
    d = Bx(a(:,:,1)) + By(a(:,:,2));
    
% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
function Fxu = Fx(u)
    [m,n] = size(u);
    Fxu = circshift(u,[0 -1])-u;

% Forward derivative operator on y with boundary condition u(m+1,:,:)=u(1,:,:)
function Fyu = Fy(u)
    [m,n] = size(u);
    Fyu = circshift(u,[-1 0])-u;

% Backward derivative operator on x with boundary condition Bxu(:,1)=u(:,1)
function Bxu = Bx(u)
    [~,n] = size(u);
    Bxu = u - circshift(u,[0 1]);

% Backward derivative operator on y with boundary condition Bxu(1,:)=u(1,:)
function Byu = By(u)
    [m,~] = size(u);
    Byu = u - circshift(u,[1 0]);

%Constructing the test image
function [F,D] = create_square_image(N,sparsity)
    %build an image of a square
    image = zeros(N,N);
    image(N/4:3*N/4,N/4:3*N/4)=255;
    
    %image = imread('card1.bmp');
    image = imread('MR_Toy_example.JPG');
    %image = imread('smooth_denoised.png');
    image = double(image(:,:,1));
    
    [m,n] = size(image);
    %build the sampling matrix, D
    D = rand(m,n);
    D = double(D<sparsity);

    %Creating the undersampled k-space:
    F = D.*fft2(image)/sqrt(m*n);

    %Plotting the undersampled image:
    figure;
    imagesc(abs(ifft2(F)));colormap('gray')
