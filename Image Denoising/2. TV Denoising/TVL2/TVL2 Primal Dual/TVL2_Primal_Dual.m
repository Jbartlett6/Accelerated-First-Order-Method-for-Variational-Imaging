function v = TVL2_Primal_Dual()
close all
clear
clc
%Read the noisy image.
f0 = imread('lenaNoise.bmp');
%f0 = imread('smooth1.bmp');
f0 = f0(:,:,1);
[m,n] = size(f0);
f0 = double(f0)/255;
%Plot the original noisy image
figure; imagesc(f0); colormap(gray); axis off; axis equal;

%Initialising the parameters: 
lambda = 0.2;
sigma = 1/sqrt(8*lambda^2);
tau = 1/sqrt(8*lambda^2);
prev_x = zeros(m,n);
prev_y = zeros(m,n,2);
prev_x_bar = zeros(m,n);
theta = 0;

for i = 1:1000

    
%Updating x (The Primal Variable)
%Gradiend update
h = prev_x + tau*lambda*(Bx(prev_y(:,:,1))+By(prev_y(:,:,2)));
%Proximal update
const = tau/(tau+1);
curr_x = const*((1/tau)*h+f0);    

%Updating the Velocity:
curr_x_bar = curr_x + theta*(curr_x-prev_x);

%Updating y (the Dual Variable)
%Gradient updated
g = prev_y + sigma*lambda*cat(3,Fx(curr_x_bar),Fy(curr_x_bar));
%Proximal update
g1 = g(:,:,1);
g2 = g(:,:,2);
mag = sqrt(g1.^2+g2.^2);
q = max(mag,1);
curr_y = cat(3,g1./q,g2./q);

%Calculating the duality gap:
primal_var = f0 + lambda*(Bx(curr_y(:,:,1))+By(curr_y(:,:,2)));
v(i) = Evalu(primal_var) - Dual_Eval(curr_y); 
 

%Resetting the variables:
prev_x = curr_x;
prev_y = curr_y;
prev_x_bar = curr_x_bar;


end
figure; imagesc(curr_x); colormap(gray); axis off; axis equal; 
figure; plot(log(v));



%Derivative Functions:
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

%Evaluating the primal objective function:
function Ev = Evalu(u)
source = double(imread('lenaNoise.bmp'));
lambda = 0.2;
f0 = source(:,:,1) / 255;
dx = Fx(u);
dy = Fy(u);
gradu_norm = sqrt((dx).^2+(dy).^2);
n = sum((u-f0).^2,'all');
Ev = 0.5*n+lambda*sum(gradu_norm(:));

%Evaluating the Dual Objective Function
function duev = Dual_Eval(u)
source = double(imread('lenaNoise.bmp'));
lambda = 0.2;
f0 = source(:,:,1) / 255;

div = Bx(u(:,:,1))+By(u(:,:,2));
first = -lambda*sum(div.*f0,'all');

div_norm = sum(div.*div,'all');

second = -((lambda^2)/2)*div_norm;
duev = first + second;



