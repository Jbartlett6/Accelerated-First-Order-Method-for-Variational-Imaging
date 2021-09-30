%ADMM FFT
function u = ADMM_FFT()
close all
clear 

%Loading the Image
source = double(imread('lenaNoise.bmp'));
%source = double(imread('smooth1.bmp'));;
f0 = source(:,:,1) / 255;
[m, n] = size(f0);
%f0 = f0(:);

%Initialisation 
rho = 3  ;
prev_v = zeros(m,n,2);
prev_u = zeros(m,n,1);
prev_mu = zeros(m,n,2);
lambda = 0.2;
%I = speye(m,n);
alpha = 1.8;

%Creating G matrix:
[Y, X] = meshgrid(0:n-1,0:m-1);
G = 2*cos(2*pi*X/m)+2*cos(2*pi*Y/n)-4;

tic
for i = 1:1000
    %u update using the fourier inverse method
    curr_u = real(ifft2(fft2(f0 - rho*(Bx(prev_v(:,:,1)-prev_mu(:,:,1))+By(prev_v(:,:,2)-prev_mu(:,:,2))))./(1-rho*G)));

    %v update using soft thresholding:
    grad_u = cat(3,Fx(curr_u),Fy(curr_u));
    c = grad_u+prev_mu;
    curr_v = soft_threshold(c,lambda/rho);
    
    %mu update
    curr_mu = prev_mu+grad_u-curr_v;
 
    %Evaluating the energy of the function
    u(i+1) = sum(abs(curr_u(:)-prev_u(:)))/sum(abs(curr_u(:)));
    %A = 0.5*(curr_u - f0).^2 + rho * (abs(Fx(curr_u)) + abs(Fy(curr_u)));
    %en(i) = sum(A(:));
 
    %Updating the variables for the next iteration
    prev_u = curr_u;
    prev_v = curr_v;
    prev_mu = curr_mu;
end
toc
figure; imagesc(curr_u); colormap(gray); axis off; axis equal; 
figure; plot(log(u));


%Defining the forward and backward derivatives which have been modified for
%fft
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

%Modified Soft Threshold:
function sf = soft_threshold(u,t)
    %Soft thresholding function
    u1 = u(:,:,1);
    u2 = u(:,:,2);
    norm = sqrt(u1.^2+u2.^2+eps);
    sf = (1./(norm)).*u.*max(norm-t,0);