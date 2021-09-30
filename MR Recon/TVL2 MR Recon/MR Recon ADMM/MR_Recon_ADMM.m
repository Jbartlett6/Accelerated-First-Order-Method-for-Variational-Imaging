function diff_1 = MR_Recon_ADMM()
close all
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Creating the test image
%Initialising the test image using the test_mrics template:
N = 128;
sparsity = 0.25;
% build an image of a square
%image = zeros(N,N);
%image(N/4:3*N/4,N/4:3*N/4)=255;

%Include image:
%image = imread('card1.bmp');
image = imread('MR_Toy_example.JPG');
%image = imread('smooth_denoised.png');
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APPA Algorithm
[m,n] = size(F);
scale = sqrt(n*m);
%Initialising the parameters
lambda = 0.1;
rho = 0.1;
gamma = 0.1/1000;

%Intialising the variables for the loop:
p1 = zeros(m,n);
p2 = zeros(m,n);
b1 = zeros(m,n);
b2 = zeros(m,n);
u = zeros(m,n);
prev_u = zeros(m,n);
%Initialising kernels for the script

val = lambda*ifft2((conj(D).*F))*scale;

uker = zeros(m,n);
uker(1,1) = 4;uker(1,2)=-1;uker(2,1)=-1;uker(m,1)=-1;uker(1,n)=-1;

denom = lambda*(conj(D).*D)+rho*fft2(uker)+gamma;
for i = 1:50
    %Updating M using fft inverse method
    num = val - rho*Fx(p1-b1) - rho*Fy(p2-b2)+gamma*u;
    u = ifft2(fft2(num)./denom);
    %Updating p using soft thresholding
    [p1,p2] = soft_threshold(Bx(u)+b1,By(u)+b2,1/rho);
    %Updating b using simple update formula
    b1 = b1 + Bx(u) - p1;
    b2 = b2 + By(u) - p2;
    
    res = u - prev_u;
    diff_1(i) = sum(abs(res(:)))/sum(abs(u(:)));
    prev_u = u;
end
figure; plot(log(diff_1))
figure;imagesc(abs(u)); colormap('gray');

    
function Fxu = Fx(u)
Fxu = circshift(u,[0 -1])-u;



% Forward derivative operator on y with boundary condition u(m+1,:,:)=u(1,:,:)
function Fyu = Fy(u)
Fyu = circshift(u,[-1 0])-u;

% Backward derivative operator on x with boundary condition Bxu(:,1)=u(:,1)
function Bxu = Bx(u)
Bxu = u - circshift(u,[0 1]);


% Backward derivative operator on y with boundary condition Bxu(1,:)=u(1,:)
function Byu = By(u)
Byu = u - circshift(u,[1 0]);

function [sf1,sf2] = soft_threshold(u1,u2,t)
    %Soft thresholding function
    %u1 = u(:,:,1);
    %u2 = u(:,:,2);
    norm = sqrt(u1.*conj(u1)+u2.*conj(u2)+eps);
    sf1 = u1.*(1./(norm)).*max(norm-t,0);
    sf2 = u2.*(1./(norm)).*max(norm-t,0);
    %sf = (1./(norm)).*u.*max(norm-t,0);

    
    function d = div(a)
    d = Bx(a(:,:,1)) + By(a(:,:,2));

function g = grad(a)
    g = cat(3,Fx(a),Fy(a));
    

