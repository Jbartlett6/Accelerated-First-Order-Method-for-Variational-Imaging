function energy = TSVL2_OF_Prox_Grad()
close all
clear all
clc

%A = double(imread('yos9.tif'));
%B = double(imread('yos10.tif'));
A = double(imread('card1.bmp'));
B = double(imread('card2.bmp'));
%A = imresize(A,0.5);
%B = imresize(B,0.5);

A = A(:,:,1)/255;
B = B(:,:,1)/255;

figure; imagesc(A); colormap(gray); title('original source image'); axis off; axis equal;
figure; imagesc(B); colormap(gray); title('original target image'); axis off; axis equal;


[m,n] = size(A);
 

%Calculating the derivatives which will be used in the data term:
[Ix, Iy] = computeDerivatives(A);
It = A - B;
J11 = Ix.*Ix;
J22 = Iy.*Iy;

alpha = 0.003;
beta = alpha*2e2;
gamma = 1;

prev_u = zeros(m,n);
prev_v = zeros(m,n);
u_velocity = zeros(m,n);
v_velocity = zeros(m,n);
prev_theta = 1;

for i = 1:200
     %Updating the momentum term for m_hat
    curr_theta = (1 + sqrt(1+4*prev_theta^2))/2;
    %momentum = (prev_theta - 1)/curr_theta;
    momentum = (1 - 1)/curr_theta;
    u_step = prev_u + momentum*u_velocity;
    v_step = prev_v + momentum*v_velocity;
    
    curr_u = u_step - Ix.*(Ix.*u_step + Iy.*v_step + It);
    curr_v = v_step - Iy.*(Ix.*u_step + Iy.*v_step + It);
    %[curr_u, p11, p12] = dual_nesterov_acceleration_tsv_innerloop(curr_u, alpha, beta, gamma);
    %[curr_v, p21, p22] = dual_nesterov_acceleration_tsv_innerloop(curr_v, alpha, beta, gamma);
    [curr_u, curr_v, p11, p12, p21, p22] = dual_nesterov_acceleration_tsv_innerloop_isotropic(curr_u, curr_v, alpha, beta, gamma,20);
    %Restart Decision for outer loop
    u_gen_grad = u_step(:) - curr_u(:);
    v_gen_grad = v_step(:) - curr_v(:);
    dot2 = u_gen_grad.*u_velocity(:) + v_gen_grad.*v_velocity(:); 
    %if sum(dot2,'all') >0; 
        %curr_theta = 1;
    %end
    
    energy(i) = Primal_Eval_TSV(curr_u, curr_v, p11, p12, p21, p22, Ix, Iy, It, alpha, beta, gamma);
    
    u_velocity = curr_u - prev_u;
    v_velocity = curr_v - prev_v;
    prev_theta = curr_theta;
    prev_u = curr_u;
    prev_v = curr_v;
    
end    
%visualization
figure; imagesc(A); colormap(gray); title('DVF'); axis off; axis equal;
hold on;
opflow = opticalFlow(curr_u,curr_v);
plot(opflow, 'DecimationFactor',[3 3],'ScaleFactor',10);
q = findobj(gca,'type','Quiver');
q.Color = 'r';
q.LineWidth = 1;

figure; imagesc(flowToColor(curr_u,curr_v)); title('HSV'); axis off; axis equal;
figure; plot(log(energy));
    
    
function [ux, uy]=computeDerivatives(u)
[m,n]=size(u);
C1 = circshift(u,[0 -1]); C1(:,n) = C1(:,n-1);
C2 = circshift(u,[0 1]);  C2(:,1) = C2(:,2);
C3 = circshift(u,[-1 0]); C3(m,:) = C3(m-1,:);
C4 = circshift(u,[1 0]);  C4(1,:) = C4(2,:);
ux=(C1-C2)/2;
uy=(C3-C4)/2;

%The partial difference equations
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


function EV = Primal_Eval_TSV(u, v, p11, p12, p21, p22, Ix, Iy, It, alpha, beta, gamma)
    %Anisotropic    
    %A = 0.5*(Ix.*u + Iy.*v + It).^2                     ...
    %+ alpha * (abs(Fx(u) - p11) + abs(Fy(u) - p12))     ...
    %+ alpha * (abs(Fx(v) - p21) + abs(Fy(v) - p22))     ...
    %+ 0.5*beta * (Fx(p11).^2 + Fy(p12).^2)              ...
    %+ 0.5*beta * (Fx(p21).^2 + Fy(p22).^2)              ...
    %+ 0.5*gamma*(p11.^2 + p12.^2)                       ...
    %+ 0.5*gamma*(p21.^2 + p22.^2);
    
    %Isotropic
    A = 0.5*(Ix.*u + Iy.*v + It).^2                     ...
    + alpha * sqrt(((Fx(u) - p11).^2 + (Fy(u) - p12).^2      ...
    + (Fx(v) - p21).^2 + (Fy(v) - p22).^2))              ...
    + 0.5*beta * (Fx(p11).^2 + Fy(p12).^2)              ...
    + 0.5*beta * (Fx(p21).^2 + Fy(p22).^2)              ...
    + 0.5*gamma*(p11.^2 + p12.^2)                       ...
    + 0.5*gamma*(p21.^2 + p22.^2);
    


    EV = sum(sum(A));